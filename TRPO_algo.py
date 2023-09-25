"""
TRPO algorithm.
"""
import copy

import numpy as np
import torch.optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

from base_declaration import DEVICE, DTYPE, EPS
from buffer import Episode
from continuous_policy import ContinuousPolicyNormal
from continuous_critic import ContinuousValueCritic
from tools import init_weight, discounted_rewards


class TRPO:
    def __init__(self, state_dim, action_dim, hidden_dim, max_action,
                 lmbda=0.9, kl_constraint=0.00005, alpha=0.5, critic_lr=0.01, gamma=0.9, device=DEVICE, eps=EPS):
        self.state_dim, self.action_dim, self.hidden_dim = state_dim, action_dim, hidden_dim
        self.max_action = max_action

        self.lmbda, self.kl_constraint, self.alpha, self.critic_lr, self.gamma = lmbda, kl_constraint, alpha, critic_lr, gamma

        self.device, self.eps = device, eps

        self.policy = ContinuousPolicyNormal(self.state_dim, self.action_dim, self.max_action, self.hidden_dim,
                                             "relu").to(self.device)
        self.policy = init_weight(self.policy)

        self.critic = ContinuousValueCritic(self.state_dim, self.action_dim, self.max_action, self.hidden_dim,
                                            "relu").to(self.device)
        self.critic = init_weight(self.critic)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_lr)

        self.replay_buffer = Episode(requires_distribution=False, requires_goal=False)

        print("TRPO: state_dim={}, action_dim={}, hidden_dim={}, critic_lr={}, gamma={}".format(
            self.state_dim, self.action_dim, self.hidden_dim, self.critic_lr, self.gamma))

        print("lmbda={}, kl_constraint={}, alpha={}".format(self.lmbda, self.kl_constraint, self.alpha))

    def action(self, state):
        state = torch.as_tensor(state, dtype=DTYPE, device=self.device)
        # No deterministic
        mean, logstd = self.policy(state)
        dist = Normal(mean, logstd.exp())
        action = dist.sample()
        return action.data.cpu().numpy(), None

    def store(self, state, action, reward, next_state, done, dist):
        self.replay_buffer.add(state, action, reward, next_state, done, dist)

    def hessian_matrix_vector_product(self, states, old_action_dists, vector, damping=0.1):
        mean, logstd = self.policy(states)
        new_action_dists = Normal(mean, logstd.exp())
        kl = torch.mean(kl_divergence(old_action_dists, new_action_dists))

        # Calculate First Order Gradient.
        kl_grad_first = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        kl_grad_first_vector = torch.cat([grad.view(-1) for grad in kl_grad_first])
        kl_grad_first_vector_prod = torch.dot(kl_grad_first_vector, vector)

        kl_grad_second = torch.autograd.grad(kl_grad_first_vector_prod, self.policy.parameters())
        kl_grad_second_vector = torch.cat([grad.contiguous().view(-1) for grad in kl_grad_second])

        return kl_grad_second_vector + vector * damping

    def conjugate_gradient(self, grad, old_action_dists, states):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        r_dot_r = torch.dot(r, r)

        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = r_dot_r / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / r_dot_r
            p = r + beta * p
            r_dot_r = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantages, old_log_probs, actor):
        mean, logstd = actor(states)
        action_dists = Normal(mean, logstd.exp())
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantages)

    def line_search(self, states, actions, advantages, old_log_probs, old_action_dists, max_vec):
        old_para = parameters_to_vector(self.policy.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantages, old_log_probs, self.policy)

        for i in range(15):
            # Here is the pow()!!!
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.policy)
            vector_to_parameters(new_para, new_actor.parameters())
            mean, logstd = new_actor(states)
            new_action_dists = Normal(mean, logstd.exp())
            kl_div = torch.mean(kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantages, old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para

        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                   old_log_probs, self.policy)
        grads = torch.autograd.grad(surrogate_obj, self.policy.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 用共轭梯度法计算x = H^(-1)g
        descent_direction = self.conjugate_gradient(obj_grad, old_action_dists,
                                                    states)

        Hd = self.hessian_matrix_vector_product(states, old_action_dists,
                                                descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint /
                              (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_probs,
                                    old_action_dists,
                                    descent_direction * max_coef)  # 线性搜索
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.policy.parameters())  # 用线性搜索后的参数更新策略

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def update(self):
        # Get datas
        sp = self.replay_buffer.obtain_dict()
        ## Data transition to tensor
        states = torch.tensor(sp["state"], dtype=DTYPE, device=self.device).view((-1, self.state_dim))
        actions = torch.tensor(sp["action"], dtype=DTYPE, device=self.device).view((-1, self.action_dim))
        next_states = torch.tensor(sp["next_state"], dtype=DTYPE, device=self.device).view((-1, self.state_dim))
        dones = torch.tensor(sp["done"], dtype=DTYPE, device=self.device).view((-1, 1))
        rewards = torch.tensor(sp["reward"], dtype=DTYPE, device=self.device).view((-1, 1))

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lmbda,
                                           td_delta.cpu()).to(self.device)
        mu, log_std = self.policy(states)
        old_action_dists = torch.distributions.Normal(mu.detach(),
                                                      log_std.exp().detach())
        old_log_probs = old_action_dists.log_prob(actions)
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.policy_learn(states, actions, old_action_dists, old_log_probs,
                          advantage)

        return critic_loss.data.cpu().item()
