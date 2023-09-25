"""
SAC algorithm.
"""
import numpy as np
import torch
import copy
from torch.distributions.normal import Normal
import torch.nn.functional as F

from base_declaration import *
from buffer import HindsightReplayBuffer
from continuous_policy import ContinuousPolicyNormal
from continuous_critic import ContinuousCritic


class SACHER:
    def __init__(self, state_dim, action_dim, hidden_dim, max_action, env, goal_dim, activate="relu",
                 actor_lr=0.0003, critic_lr=0.003, alpha_lr=0.005, her_ratio=0.8,
                 target_entropy=5, tau=0.005, gamma=0.98,
                 maxlen=10000, batch_size=64,
                 device=DEVICE):
        # TODO:Complete this algorithm.
        self.state_dim, self.action_dim, self.hidden_dim, self.max_action = state_dim, action_dim, hidden_dim, max_action
        self.goal_dim = goal_dim
        self.activate = activate
        self.env = env
        self.actor_lr, self.critic_lr, self.alpha_lr, self.her_ratio = actor_lr, critic_lr, alpha_lr, her_ratio
        self.target_entropy, self.tau, self.gamma = target_entropy, tau, gamma
        self.maxlen, self.batch_size = maxlen, batch_size
        self.device = device

        # Initialize the networks
        self.actor = ContinuousPolicyNormal(self.state_dim + self.goal_dim, self.action_dim, self.max_action,
                                            self.hidden_dim, self.activate, self.device).to(self.device)
        self.critic_1 = ContinuousCritic(self.state_dim + self.goal_dim, self.action_dim, self.max_action,
                                         self.hidden_dim, self.activate).to(self.device)
        self.critic_2 = ContinuousCritic(self.state_dim + self.goal_dim, self.action_dim, self.max_action,
                                         self.hidden_dim, self.activate).to(self.device)
        self.target_critic_1 = ContinuousCritic(self.state_dim + self.goal_dim, self.action_dim, self.max_action,
                                                self.hidden_dim, self.activate).to(self.device)
        self.target_critic_2 = ContinuousCritic(self.state_dim + self.goal_dim, self.action_dim, self.max_action,
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
        self.replay_buffer = HindsightReplayBuffer(self.maxlen, True, DTYPE, self.device)

    def action(self, state, train=False):
        mean, log_std = self.actor(state)
        dist = Normal(mean, log_std.exp())
        # u = pi(a|s)
        u = dist.rsample()
        # a = tanh(u)
        action = torch.tanh(u)
        # Calculate the log probs
        log_prob = dist.log_prob(u) - torch.log(1 - torch.tanh(action).pow(2) + EPS)
        if not train:
            return action.squeeze(0).data.cpu().numpy(), log_prob
        else:
            return action, log_prob

    def store(self, state, action, reward, next_state, done):
        # Actually, the 'dist' is the log_prob
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        # Sample from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        u_states, u_actions, u_rewards, u_next_states, u_dones = copy.copy(states), copy.copy(actions), copy.copy(
            rewards), copy.copy(next_states), copy.copy(dones)

        u_actions = torch.as_tensor(list(u_actions), device=self.device, dtype=DTYPE).view((-1, self.action_dim))
        u_rewards = torch.as_tensor(list(u_rewards), device=self.device, dtype=DTYPE).view((-1, 1))

        # TODO: Calculate the Q targets
        # Get s_{t+1}, log pi(a_{t+1}|s_{t+1})
        next_o_g = torch.tensor(
            [np.concatenate((i["observation"], i["desired_goal"])).tolist() for i in u_next_states],
            dtype=DTYPE, device=self.device
        )
        next_actions, next_log_prob = self.action(next_o_g, train=True)

        entropy = -1 * torch.mean(next_log_prob, dim=-1, keepdim=True)

        next_s_a = torch.cat((next_o_g, next_actions), dim=-1)
        q1 = self.target_critic_1(next_s_a)
        q2 = self.target_critic_2(next_s_a)
        q = torch.min(q1, q2) + self.log_value.exp() * entropy

        # td_target = rewards + self.gamma * q * (1 - dones)
        td_target = u_rewards + self.gamma * q

        # TODO: Update the 2 Q critic networks
        o_g = torch.tensor(
            [np.concatenate((i["observation"], i["desired_goal"])).tolist() for i in u_states],
            dtype=DTYPE, device=self.device
        )
        current_s_a = torch.cat((o_g, u_actions), dim=-1)
        critic_loss_1 = torch.mean(F.mse_loss(self.critic_1(current_s_a), td_target.detach()))
        critic_loss_2 = torch.mean(F.mse_loss(self.critic_2(current_s_a), td_target.detach()))

        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss_1.backward()
        critic_loss_2.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        # TODO: Update the actor network

        new_a, new_log_prob = self.action(o_g, train=True)
        new_entropy = -1 * new_log_prob
        new_c_s_a = torch.cat((o_g, new_a), dim=-1)
        q1_v = self.critic_1(new_c_s_a)
        q2_v = self.critic_2(new_c_s_a)

        actor_loss = torch.mean(-1 * self.log_value.exp() * new_entropy - torch.min(q1_v, q2_v)) + 0.5 * new_a.pow(
            2).mean()
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

    def her(self):
        """

        :param buffer: is used to HER.
        :param reward_function: is used to compute the reward function after HER.
        :param done_function: is used to check the transition satisfy the terminal.
        :return:
        """
        # 计算整个episode的长度
        el = len(self.replay_buffer.episode_buffer)

        # 遍历一个episode里面的每一个小transition
        for idx in range(el):

            # 如果当前transition是在self.her_ratio比率里面，那么就进行her
            if np.random.rand() < self.her_ratio:

                # 取当前的transition里面的元素展开
                i_state, i_action, i_reward, i_next_state, i_done = self.replay_buffer.episode_buffer[idx]

                new_state, new_action, new_next_state = copy.copy(i_state), copy.copy(i_action), copy.copy(i_next_state)

                # 寻找当前transition后面的设置goal的transition，将其编号设置为index
                if idx + 1 == el:
                    index = idx
                else:
                    index = np.random.randint(idx + 1, el)

                # 取出index里面transition的状态信息，抽取里面的achieved_goal作为目标
                index_state, _1, _2, _3, _4 = self.replay_buffer.episode_buffer[index]

                index_state_copy = copy.copy(index_state)

                if self.replay_buffer.state_is_dict:
                    # Create a new goal using the achieved goal from the next state
                    new_goal = index_state_copy['achieved_goal']

                    # Update the state and next_state with the new goal
                    new_state['desired_goal'] = new_goal
                    new_next_state['desired_goal'] = new_goal

                    # Calculate the new reward based on the updated state and next_state
                    new_reward = \
                        self.env.env.compute_reward(new_state["achieved_goal"],
                                                    new_state['desired_goal'], {})

                    # Check the done is True or not
                    # new_done = True if self.goal_distance(i_state["achieved_goal"], i_state[
                    #     "desired_goal"]) < self.env.env.distance_threshold else False

                    # Store the HER experience
                    self.replay_buffer.add(new_state, new_action, new_reward, new_next_state, i_done)
            else:
                pass

        self.replay_buffer.add_to_buffer(self.replay_buffer.episode_buffer)
        self.replay_buffer.clear()

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
