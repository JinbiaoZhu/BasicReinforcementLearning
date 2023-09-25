"""
Distributed PPO algorithm.
"""
import random

import numpy as np
from typing import Dict
from base_declaration import *
from tools import init_weight, set_seed
from buffer import Episode
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch.multiprocessing import Process, Pipe
import gym
from evaluation import EpisodeEvaluate
import os


class ProcessPPO:
    """
    This is a PPO's implementation in a subprocess.
    """

    def __init__(self, state_dim, action_dim, hidden_dim, max_action,
                 actor_lr, critic_lr, gamma, lmbda, epsilon,
                 epochs, device):

        self.state_dim, self.action_dim, self.hidden_dim = state_dim, action_dim, hidden_dim
        self.max_action = max_action

        self.actor_lr, self.critic_lr = actor_lr, critic_lr
        self.gamma, self.lmbda, self.epsilon = gamma, lmbda, epsilon

        self.epochs, self.device = epochs, device

        self.actor, self.critic = None, None

        self.replay_buffer = Episode(requires_distribution=False, requires_goal=False)

    def connect(self, net, is_init=True):
        self.actor = init_weight(net[0]) if is_init else net[0]
        self.critic = init_weight(net[1]) if is_init else net[1]
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_lr)

    def action(self, state):
        state = torch.from_numpy(state).to(dtype=DTYPE, device=self.device)
        # No deterministic
        mean, logstd = self.actor(state)
        dist = Normal(mean, logstd.exp())
        action = dist.sample()
        return action.data.cpu().numpy()

    def store(self, state, action, reward, next_state, done, _):
        self.replay_buffer.add(state, action, reward, next_state, done, _)

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().cpu().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.from_numpy(np.asarray(advantage_list)).to(dtype=DTYPE, device=self.device)

    def update(self):
        # Get datas
        sp = self.replay_buffer.obtain_dict()
        # Data transition to tensor
        states = torch.from_numpy(sp["state"]).to(dtype=DTYPE, device=self.device).view((-1, self.state_dim))
        actions = torch.from_numpy(sp["action"]).to(dtype=DTYPE, device=self.device).view((-1, self.action_dim))
        next_states = torch.from_numpy(sp["next_state"]).to(dtype=DTYPE, device=self.device).view((-1, self.state_dim))
        dones = torch.from_numpy(sp["done"]).to(dtype=DTYPE, device=self.device).view((-1, 1))
        rewards = torch.from_numpy(sp["reward"]).to(dtype=DTYPE, device=self.device).view((-1, 1))

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta)

        mu, log_std = self.actor(states)
        old_action_dists = torch.distributions.Normal(mu.detach(),
                                                      log_std.exp().detach())
        old_log_probs = old_action_dists.log_prob(actions)

        actor_loss_total, critic_loss_total = 0, 0

        for i in range(self.epochs):
            mu, log_std = self.actor(states)
            action_dists = Normal(mu, log_std.exp())
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            actor_loss = -1 * torch.mean(torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            actor_loss_total += actor_loss.data.item()
            critic_loss_total += critic_loss.data.item()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return actor_loss_total, critic_loss_total


def child_func(pipe, config, queue, shared_list, env):
    while True:
        net = pipe.recv()
        eval = EpisodeEvaluate()
        process_ppo = ProcessPPO(config["state_dim"], config["action_dim"], config["hidden_dim"], config["max_action"],
                                 config["actor_lr"], config["critic_lr"],
                                 config["gamma"], config["lmbda"], config["epsilon"],
                                 config["epoch"], config["device"])
        process_ppo.connect(net, is_init=False)

        eval.episode_return_is_zero()

        # Obtain an episode
        state = env.reset()
        done = False

        while not done:
            action = process_ppo.action(state)
            next_state, reward, done, _ = env.step(action)
            process_ppo.store(state, action, reward, next_state, done, _)
            state = next_state
            eval.add_single_step_reward(reward)

        actor_loss, critic_loss = process_ppo.update()
        total_reward = eval.return_total_reward()

        net[0], net[1] = net[0].to("cpu"), net[1].to("cpu")
        info = [net[0], net[1], total_reward, actor_loss, critic_loss]
        shared_list.append(info)
        queue.put(shared_list)

        del eval, process_ppo
