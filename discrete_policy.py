"""
Discrete policies, used for DQN, DoubleDQN, DuelingDQN.
"""
import torch
import torch.nn as nn

from base_network import BaseNet


class DiscreteQNet(BaseNet):
    def __init__(self, state_dim, action_dim, hidden_dim=256, activate="relu"):
        super().__init__()

        if activate in ["relu", "tanh"]:
            if activate == "relu":
                self.activate = nn.ReLU()
            elif activate == "tanh":
                self.activate = nn.Tanh()
            else:
                raise ValueError("Unknown activation function.")

        self.q1 = nn.Linear(state_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, hidden_dim)
        self.q3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.activate(self.q1(x))
        x = self.activate(self.q2(x))
        return self.q3(x)


class DiscreteVANet(BaseNet):
    def __init__(self, state_dim, action_dim, hidden_dim=256, activate="relu"):
        super().__init__()

        if activate in ["relu", "tanh"]:
            if activate == "relu":
                self.activate = nn.ReLU()
            elif activate == "tanh":
                self.activate = nn.Tanh()
            else:
                raise ValueError("Unknown activation function.")

        # Share net q1
        self.q1 = nn.Linear(state_dim, hidden_dim)
        # Value net belong q2
        self.q2_V = nn.Linear(hidden_dim, 1)
        # Advantage net belong q3
        self.q3_A = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # for A
        A = self.q3_A(self.activate(self.q1(x)))
        A -= torch.mean(A, dim=1).view(-1, 1)

        # for V
        V = self.q2_V(self.activate(self.q1(x)))

        # for Q
        Q = V + A
        return Q
