"""
A2C algo.
"""
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from base_declaration import *
from continuous_policy import ContinuousPolicyNormal
from continuous_critic import ContinuousValueCritic
from tools import init_weight
from buffer import SequenceReplayBuffer


class A2C:
    def __init__(self, state_dim, action_dim, max_action,
                 hidden_dim, activate="relu",
                 actor_lr=0.0003, critic_lr=0.003, gamma=0.98,
                 maxlen=10000, batch_size=64, requires_dist=True,
                 device=DEVICE):
        self.state_dim, self.action_dim, self.max_action = state_dim, action_dim, max_action
        self.hidden_dim, self.activate = hidden_dim, activate
        self.actor_lr, self.critic_lr, self.gamma = actor_lr, critic_lr, gamma
        self.maxlen, self.batch_size, self.requires_dist = maxlen, batch_size, requires_dist
        self.device = device

        self.actor = ContinuousPolicyNormal(self.state_dim, self.action_dim, self.max_action,
                                            self.hidden_dim, self.activate).to(self.device)
        self.actor = init_weight(self.actor)

        self.critic = ContinuousValueCritic(self.state_dim, self.action_dim, self.max_action,
                                            self.hidden_dim, self.activate).to(self.device)
        self.critic = init_weight(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_lr)

        self.buffer = SequenceReplayBuffer(self.maxlen, self.batch_size, self.requires_dist)

        self.temp = 1.0

        print("A2C: state_dim={}, action_dim={}, hidden_dim={}, lr={}, gamma={}".format(
            self.state_dim, self.action_dim, self.hidden_dim, [self.actor_lr, self.critic_lr], self.gamma))

        print("maxlen={}, batch_size={}".format(self.maxlen, self.batch_size))

    def action(self, state):
        mean, logstd = self.actor(state)
        std = logstd.exp()
        dist = Normal(mean, std)
        return dist.sample().data.cpu().numpy(), dist

    def store(self, state, action, reward, next_state, done, dist):
        self.buffer.add_to_episode(state, action, reward, next_state, done, dist)

    def update(self):
        # Sample from the replay buffer
        sp = self.buffer.get_from_episode()

        # Data transition
        states = torch.tensor(sp["state"], dtype=DTYPE, device=self.device)
        actions = torch.tensor(sp["action"], dtype=DTYPE, device=self.device).view((-1, self.action_dim))
        rewards = torch.tensor(sp["reward"], dtype=DTYPE, device=self.device).view((-1, 1))
        next_states = torch.tensor(sp["next_state"], dtype=DTYPE, device=self.device)
        dones = torch.tensor(sp["done"], dtype=DTYPE, device=self.device).view((-1, 1))

        # Preparation
        current_s_v, next_s_v = self.critic(states), self.critic(next_states)

        # Version 1
        # Calculate temporal difference target
        td_target = rewards + self.gamma * (1 - dones) * next_s_v

        # Calculate critic loss
        critic_loss = torch.mean(F.mse_loss(current_s_v, td_target))

        # Calculate actor's loss
        means, log_std = self.actor(states)
        dists = Normal(means, log_std.exp())
        log_probs = dists.log_prob(actions)
        delta = td_target - current_s_v
        actor_loss = torch.mean(-1 * self.temp * log_probs * delta.detach())

        # Optimize the two loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Clear the replay buffer
        self.buffer.clear_episode()

        return actor_loss.item(), critic_loss.item()
