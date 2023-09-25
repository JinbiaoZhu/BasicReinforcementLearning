"""
RANDOM algorithm.
"""

from torch.distributions.normal import Normal

from base_declaration import *

from buffer import SequenceReplayBuffer



class RANDOM:
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

        self.buffer = SequenceReplayBuffer(self.maxlen, self.batch_size, self.requires_dist)

        print("RANDOM: state_dim={}, action_dim={}, hidden_dim={}, lr={}, gamma={}".format(
            self.state_dim, self.action_dim, self.hidden_dim, [self.actor_lr, self.critic_lr], self.gamma))

        print("maxlen={}, batch_size={}".format(self.maxlen, self.batch_size))

    def action(self, state):
        """
        Randomly choose the action in the action space.
        The distribution returned is the standard normal distribution.
        :param state:
        :return:
        """
        mean, std = torch.zeros(self.action_dim), torch.ones(self.action_dim)
        dist = Normal(mean, std)
        action_atom = (torch.randn(self.action_dim) - 0.5) * 2 - 1
        return action_atom.data.cpu().numpy(), dist

    def store(self, state, action, reward, next_state, done, dist):
        pass

    def update(self):
        pass
