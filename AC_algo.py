"""
AC algorithm.
"""
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

from base_declaration import *
from tools import init_weight
from buffer import SequenceReplayBuffer
from continuous_critic import ContinuousValueCritic
from continuous_policy import ContinuousPolicyNormal


class AC:
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

        print("AC: state_dim={}, action_dim={}, hidden_dim={}, lr={}, gamma={}".format(
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
        dists = sp["dist"]

        # Preparation
        current_states_values, next_states_values = self.critic(states), self.critic(next_states)
        td_target = rewards + self.gamma * next_states_values * (1 - dones)
        delta = td_target - current_states_values

        # Version 1
        # Calculate temporal difference target
        critic_loss = torch.mean(F.mse_loss(current_states_values, td_target.detach()))

        # # Version 2
        # critic_loss = torch.mean(-1 * delta.detach() * current_states_values)

        # Calculate actor's loss
        mean, logstd = self.actor(states)
        dists = Normal(mean, torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX).exp())
        log_probs = dists.log_prob(actions)
        actor_loss = torch.mean(-1 * log_probs * delta.detach())

        # Optimize this loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.buffer.clear_episode()

        return critic_loss.item(), actor_loss.item()
