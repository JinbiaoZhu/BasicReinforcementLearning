"""
REINFORCE algorithm.
"""
import numpy as np
import torch.optim
from torch.distributions.normal import Normal

from base_declaration import DEVICE, DTYPE, EPS
from buffer import SequenceReplayBuffer
from continuous_policy import ContinuousPolicyNormal
from tools import init_weight, discounted_rewards


class REINFORCE:
    def __init__(self, state_dim, action_dim, hidden_dim, max_action,
                 lr=0.01, gamma=0.99,
                 max_len=10000, batch_size=64,
                 device=DEVICE, eps=EPS):
        self.state_dim, self.action_dim, self.hidden_dim = state_dim, action_dim, hidden_dim
        self.max_action = max_action
        self.lr, self.gamma = lr, gamma
        self.max_len, self.batch_size = max_len, batch_size
        self.device, self.eps = device, eps

        self.policy = ContinuousPolicyNormal(self.state_dim, self.action_dim, self.max_action, self.hidden_dim,
                                             "relu").to(self.device)
        self.policy = init_weight(self.policy)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.replay_buffer = SequenceReplayBuffer(self.max_len, self.batch_size, True)

        print("REINFORCE: state_dim={}, action_dim={}, hidden_dim={}, lr={}, gamma={}".format(
            self.state_dim, self.action_dim, self.hidden_dim, self.lr, self.gamma))

        print("maxlen={}, batch_size={}".format(self.max_len, self.batch_size))

    def action(self, state):
        # No deterministic
        state = torch.tensor(state, dtype=DTYPE, device=self.device)
        mean, logstd = self.policy(state)
        dist = Normal(mean, logstd.exp())
        return dist.sample().data.cpu().numpy(), dist

    def store(self, state, action, reward, next_state, done, dist):
        self.replay_buffer.add_to_episode(state, action, reward, next_state, done, dist)

    def update(self):
        # Get datas
        sp = self.replay_buffer.get_from_episode()
        ## Data transition to tensor
        states = torch.tensor(sp["state"], dtype=DTYPE, device=self.device)
        actions = torch.tensor(sp["action"], dtype=DTYPE, device=self.device)
        dists = sp["dist"]
        rewards = torch.tensor(sp["reward"], dtype=DTYPE, device=self.device).view((-1, 1))

        # Calculate discounted rewards
        d_r = discounted_rewards(rewards, self.gamma)

        # Update the policy's gradient
        self.optimizer.zero_grad()

        loss = torch.zeros((1, 1), dtype=DTYPE, device=self.device)
        for i in range(len(states)):
            s, a, dr, dist = states[i], actions[i], d_r[i], dists[i]
            mean, logstd = self.policy(s)
            a_log_prob = Normal(mean, logstd.exp()).log_prob(a)
            loss += (-1 * (a_log_prob.sum()) * d_r[i])

        loss /= len(states)
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)

        self.optimizer.step()

        self.replay_buffer.clear_episode()

        return loss.data.cpu().item()
