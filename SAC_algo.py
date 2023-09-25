"""
SAC algorithm.
"""
import numpy as np
import torch

from torch.distributions.normal import Normal
import torch.nn.functional as F

from base_declaration import *
from buffer import ReplayBuffer
from continuous_policy import ContinuousPolicyNormal
from continuous_critic import ContinuousCritic


class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim, max_action, activate="relu",
                 actor_lr=0.0003, critic_lr=0.003, alpha_lr=0.005,
                 target_entropy=5, tau=0.005, gamma=0.98,
                 maxlen=10000, batch_size=64,
                 device=DEVICE):
        # TODO:Complete this algorithm.
        self.state_dim, self.action_dim, self.hidden_dim, self.max_action = state_dim, action_dim, hidden_dim, max_action
        self.activate = activate
        self.actor_lr, self.critic_lr, self.alpha_lr = actor_lr, critic_lr, alpha_lr
        self.target_entropy, self.tau, self.gamma = target_entropy, tau, gamma
        self.maxlen, self.batch_size = maxlen, batch_size
        self.device = device

        # Initialize the networks
        self.actor = ContinuousPolicyNormal(self.state_dim, self.action_dim, self.max_action,
                                            self.hidden_dim, self.activate).to(self.device)
        self.critic_1 = ContinuousCritic(self.state_dim, self.action_dim, self.max_action,
                                         self.hidden_dim, self.activate).to(self.device)
        self.critic_2 = ContinuousCritic(self.state_dim, self.action_dim, self.max_action,
                                         self.hidden_dim, self.activate).to(self.device)
        self.target_critic_1 = ContinuousCritic(self.state_dim, self.action_dim, self.max_action,
                                                self.hidden_dim, self.activate).to(self.device)
        self.target_critic_2 = ContinuousCritic(self.state_dim, self.action_dim, self.max_action,
                                                self.hidden_dim, self.activate).to(self.device)

        # Reset the 2 critic networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Initialize the 3 optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), self.critic_lr)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), self.critic_lr)

        # Use the log value og alpha, which can make the training process more stable
        self.log_value = torch.tensor(np.log(0.01), dtype=DTYPE, device=self.device, requires_grad=True)
        self.log_value_optimizer = torch.optim.Adam([self.log_value], self.alpha_lr)

        # We don't need to store the distributions  in the replay buffer.
        self.replay_buffer = ReplayBuffer(self.maxlen, self.batch_size)

    def action(self, state):
        state = torch.as_tensor(state, dtype=DTYPE, device=self.device)
        mean, log_std = self.actor(state)
        dist = Normal(mean, log_std.exp())
        # u = pi(a|s)
        u = dist.rsample()
        # a = tanh(u)
        action = torch.tanh(u)
        # Calculate the log probs
        log_prob = dist.log_prob(u) - torch.log(1 - torch.tanh(action).pow(2) + EPS)
        return action, log_prob

    def store(self, state, action, reward, next_state, done):
        # Actually, the 'dist' is the log_prob
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        # Sample from the replay buffer
        sp = self.replay_buffer.sample()

        # Data transition
        states = torch.tensor(sp["state"], dtype=DTYPE, device=self.device)
        actions = torch.tensor(sp["action"], dtype=DTYPE, device=self.device).view((-1, self.action_dim))
        rewards = torch.tensor(sp["reward"], dtype=DTYPE, device=self.device).view((-1, 1))
        next_states = torch.tensor(sp["next_state"], dtype=DTYPE, device=self.device)
        dones = torch.tensor(sp["done"], dtype=DTYPE, device=self.device).view((-1, 1))

        # TODO: Calculate the Q targets

        # Get s_{t+1}, log pi(a_{t+1}|s_{t+1})
        next_actions, next_log_prob = self.action(next_states)

        entropy = -1 * torch.mean(next_log_prob, dim=-1, keepdim=True)

        next_s_a = torch.cat((next_states, next_actions), dim=-1)
        q1 = self.target_critic_1(next_s_a)
        q2 = self.target_critic_2(next_s_a)
        q = torch.min(q1, q2) + self.log_value.exp() * entropy

        td_target = rewards + self.gamma * q * (1 - dones)

        # TODO: Update the 2 Q critic networks
        current_s_a = torch.cat((states, actions), dim=-1)
        critic_loss_1 = torch.mean(F.mse_loss(self.critic_1(current_s_a), td_target.detach()))
        critic_loss_2 = torch.mean(F.mse_loss(self.critic_2(current_s_a), td_target.detach()))

        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss_1.backward()
        critic_loss_2.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        # TODO: Update the actor network

        new_a, new_log_prob = self.action(states)
        new_entropy = -1 * new_log_prob
        new_c_s_a = torch.cat((states, new_a), dim=-1)
        q1_v = self.critic_1(new_c_s_a)
        q2_v = self.critic_2(new_c_s_a)

        actor_loss = torch.mean(-1 * self.log_value.exp() * new_entropy - torch.min(q1_v, q2_v))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # TODO: Update the alpha
        alpha_loss = torch.mean((new_entropy - self.target_entropy).detach() * self.log_value.exp())
        self.log_value_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_value_optimizer.step()

        # TODO: Softly update the target Q network
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        return actor_loss.data.item(), critic_loss_1.data.item(), critic_loss_2.data.item(), alpha_loss.data.item()

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
