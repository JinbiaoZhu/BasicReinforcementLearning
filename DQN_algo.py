"""
DQN algorithm.
"""
import numpy as np
import torch
import torch.nn.functional as F

from base_declaration import DEVICE, DTYPE, DTYPE_int
from buffer import ReplayBuffer
from discrete_policy import DiscreteQNet
from tools import init_weight


class DQN:
    def __init__(self,
                 state_dim, action_dim, hidden_dim,
                 lr=0.001, gamma=0.99, epsilon=0.5, target_f=10,
                 max_len=10000, batch_size=128,
                 device=DEVICE):

        self._state_dim, self._action_dim, self._hidden_dim = state_dim, action_dim, hidden_dim

        self._lr, self._gamma, self._epsilon, self._target_f = lr, gamma, epsilon, target_f
        self._max_len, self._batch_size = max_len, batch_size

        print("DQN: state_dim={}, action_dim={}, hidden_dim={}, lr={}, gamma={}, epsilon={}, target_f={}".format(
            self._state_dim, self._action_dim, self._hidden_dim, lr, gamma, epsilon, target_f))

        print("maxlen={}, batch_size={}".format(self._max_len, self._batch_size))

        self._device = device

        self.DiscreteQNet = DiscreteQNet(self._state_dim, self._action_dim, self._hidden_dim).to(self._device)
        self.DiscreteQNet = init_weight(self.DiscreteQNet)

        self.target_DiscreteQNet = DiscreteQNet(self._state_dim, self._action_dim, self._hidden_dim).to(self._device)
        # self.target_DiscreteQNet.load_state_dict(self.DiscreteQNet.state_dict())

        # Version 1
        self.lossFunc = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.DiscreteQNet.parameters(), lr=self._lr)

        self.replay_buffer = ReplayBuffer(self._max_len, self._batch_size)

        self.count = 0

    def action(self, state):
        if np.random.random() < self._epsilon:
            action = np.random.randint(self._action_dim)
        else:
            state = torch.tensor(state, dtype=DTYPE, device=DEVICE).view((1, -1))
            action = self.DiscreteQNet(state).argmax().item()
        return action

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):

        # Sample from the replay buffer
        sp = self.replay_buffer.sample()

        # Data transition
        states = torch.tensor(sp["state"], dtype=DTYPE, device=DEVICE)
        actions = torch.tensor(sp["action"], dtype=DTYPE_int, device=DEVICE).view((-1, 1))
        rewards = torch.tensor(sp["reward"], dtype=DTYPE, device=DEVICE).view((-1, 1))
        next_states = torch.tensor(sp["next_state"], dtype=DTYPE, device=DEVICE)
        dones = torch.tensor(sp["done"], dtype=DTYPE_int, device=DEVICE).view((-1, 1))

        # Q-value estimation

        ## Calculate Q_{w}(s,a)
        current_q_values = self.DiscreteQNet(states).gather(1, actions)

        ## Calculate Q_{w-}(s',a')
        # next_q_values = torch.max(self.target_DiscreteQNet(next_states), dim=1)[0].view((-1, 1))
        next_q_values = self.target_DiscreteQNet(next_states).max(1)[0].view((-1, 1))

        ## Calculate y = r + max Q_{w-}(s',a')
        y = rewards + self._gamma * next_q_values * (1 - dones)

        ## Calculate loss
        ### Version 1 : Use the torch.nn.MSELoss()
        # loss = self.lossFunc(current_q_values, y)
        ### Version 2 : Use the torch.nn.functional
        loss = torch.mean(F.mse_loss(current_q_values, y))

        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self._target_f == 0:
            self.target_DiscreteQNet.load_state_dict(self.DiscreteQNet.state_dict())

        self.count += 1

        return loss.data.item()
