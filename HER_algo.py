"""
Hindsight Experience Replay, HER implementation.
Use DDPG or SAC for updating.
"""
import copy

import torch

from buffer import HindsightReplayBuffer
from base_declaration import *
from continuous_policy import ContinuousPolicyDeterministic
from continuous_critic import ContinuousCritic
import numpy as np
from tools import init_weight


class DDPG_HER:
    def __init__(self, state_dim, action_dim, hidden_dim, max_action,
                 goal_dim,
                 env,
                 activate="relu",
                 sigma=0.5, actor_lr=0.0003, critic_lr=0.003, tau=0.005, gamma=0.98,
                 maxlen=10000, batch_size=64,
                 her_ratio=0.8,
                 device=DEVICE):
        self.state_dim, self.action_dim, self.hidden_dim, self.max_action = state_dim, action_dim, hidden_dim, max_action
        self.goal_dim = goal_dim
        self.env = env
        self.activate = activate
        self.sigma, self.actor_lr, self.critic_lr, self.tau, self.gamma = sigma, actor_lr, critic_lr, tau, gamma
        self.maxlen, self.batch_size = maxlen, batch_size
        self.her_ratio = her_ratio
        self.device = device

        self.actor = ContinuousPolicyDeterministic(self.state_dim + self.goal_dim, self.action_dim, self.max_action,
                                                   self.hidden_dim,
                                                   self.activate, self.device).to(self.device)
        self.actor = init_weight(self.actor)
        self.critic = ContinuousCritic(self.state_dim + self.goal_dim, self.action_dim, self.max_action,
                                       self.hidden_dim,
                                       self.activate).to(self.device)
        self.critic = init_weight(self.critic)

        self.target_actor = ContinuousPolicyDeterministic(self.state_dim + self.goal_dim, self.action_dim,
                                                          self.max_action,
                                                          self.hidden_dim, self.activate, self.device).to(self.device)

        self.target_critic = ContinuousCritic(self.state_dim + self.goal_dim, self.action_dim, self.max_action,
                                              self.hidden_dim,
                                              self.activate).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.critic_lossfunc = torch.nn.MSELoss()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_lr)

        self.replay_buffer = HindsightReplayBuffer(self.maxlen,
                                                   self.state_dim, self.action_dim, self.goal_dim)

        print("DDPG+HER: state_dim={}, action_dim={}, hidden_dim={}, lr={}, gamma={}, sigma={}, tau={}".format(
            self.state_dim, self.action_dim, self.hidden_dim, [self.actor_lr, self.critic_lr], self.gamma, self.sigma,
            self.tau))

        print("maxlen={}, batch_size={}".format(self.maxlen, self.batch_size))

    def action(self, state):
        if np.random.rand() < self.sigma:
            action = self.actor(state).squeeze(0).data.cpu().numpy() + 0.5 * self.max_action * 2 * (
                    np.random.rand(self.action_dim) - 0.5)
        else:
            action = self.max_action * 2 * (np.random.rand(self.action_dim) - 0.5)
        return action

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        # Sample from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        actions = torch.as_tensor(list(actions), device=self.device, dtype=DTYPE).view((-1, self.action_dim))
        rewards = torch.as_tensor(list(rewards), device=self.device, dtype=DTYPE).view((-1, 1))
        # dones = torch.as_tensor(list(dones), device=self.device, dtype=DTYPE).view((-1, 1))

        # Update the critic network
        # Calculate the next q values
        # Don't add noise here !

        # Calculate the next actions
        with torch.no_grad():
            # Calculate a_{t+1} to tensor

            next_o_g = torch.tensor(
                [np.concatenate((i["observation"], i["desired_goal"])).tolist() for i in next_states],
                dtype=DTYPE, device=self.device
            )
            next_actions = self.target_actor(next_o_g)
            # next_actions = torch.tensor(next_actions, dtype=DTYPE, device=DEVICE)

            # Create the (s_{t+1},a_{t+1}) in the form of tensor
            next_input = torch.cat((next_o_g, next_actions), dim=-1)

            # Calculate next_q_values
            next_q_values = self.target_critic(next_input)
            # y = rewards + self.gamma * next_q_values * (1 - dones)
            y = rewards + self.gamma * next_q_values

        # Calculate current_q_values
        o_g = torch.tensor(
            [np.concatenate((i["observation"], i["desired_goal"])).tolist() for i in states],
            dtype=DTYPE, device=self.device
        )
        current_input = torch.cat((o_g, actions), dim=-1)
        current_q_values = self.critic(current_input)

        # Calculate the critic loss
        crtitc_loss = self.critic_lossfunc(y, current_q_values).mean()

        # Update
        self.critic_optimizer.zero_grad()
        crtitc_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        # Don't add noise here !

        # Version 1
        # current_now_actions = self.actor(states)
        # current_now_actions = torch.tensor(current_now_actions, dtype=DTYPE, device=DEVICE)
        # current_input = torch.cat((states, current_now_actions), dim=-1)
        # monitor = self.critic(current_input)
        # actor_loss = -1 * torch.mean(monitor)

        # Version 2
        a = self.actor(o_g)
        actor_loss = -1 * torch.mean(self.critic(torch.cat((o_g, a), dim=-1))) + 3 * a.pow(2).mean()
        # actor_loss = -1 * torch.mean(self.critic(torch.cat((o_g, a), dim=-1)))

        # Update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return actor_loss.data.item(), crtitc_loss.data.item()

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

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
