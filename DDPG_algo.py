"""
DDPG algorithm.
"""
import numpy as np

from base_declaration import *
from buffer import ReplayBuffer
from continuous_critic import ContinuousCritic
from continuous_policy import ContinuousPolicyDeterministic


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim, max_action, activate="relu",
                 sigma=0.001, actor_lr=0.0003, critic_lr=0.003, tau=0.005, gamma=0.98,
                 maxlen=10000, batch_size=64,
                 device=DEVICE):
        self.state_dim, self.action_dim, self.hidden_dim, self.max_action = state_dim, action_dim, hidden_dim, max_action
        self.activate = activate
        self.sigma, self.actor_lr, self.critic_lr, self.tau, self.gamma = sigma, actor_lr, critic_lr, tau, gamma
        self.maxlen, self.batch_size = maxlen, batch_size
        self.device = device

        self.actor = ContinuousPolicyDeterministic(self.state_dim, self.action_dim, self.max_action,
                                                   self.hidden_dim, self.activate).to(self.device)
        self.critic = ContinuousCritic(self.state_dim, self.action_dim, self.max_action, self.hidden_dim,
                                       self.activate).to(self.device)

        self.target_actor = ContinuousPolicyDeterministic(self.state_dim, self.action_dim, self.max_action,
                                                          self.hidden_dim, self.activate).to(self.device)
        self.target_critic = ContinuousCritic(self.state_dim, self.action_dim, self.max_action, self.hidden_dim,
                                              self.activate).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.critic_lossfunc = torch.nn.MSELoss()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_lr)

        self.replay_buffer = ReplayBuffer(self.maxlen, self.batch_size)

        print("DDPG: state_dim={}, action_dim={}, hidden_dim={}, lr={}, gamma={}, sigma={}, tau={}".format(
            self.state_dim, self.action_dim, self.hidden_dim, [self.actor_lr, self.critic_lr], self.gamma, self.sigma,
            self.tau))

        print("maxlen={}, batch_size={}".format(self.maxlen, self.batch_size))

    def action(self, state):
        state = torch.tensor(state, dtype=DTYPE, device=self.device)
        action = self.actor(state).data.cpu().numpy() + self.max_action * self.sigma * 2 * (
                np.random.rand(self.action_dim) - 0.5)
        return action

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update(self):
        # Sample from the replay buffer
        sp = self.replay_buffer.sample()

        # Data transition
        states = torch.tensor(sp["state"], dtype=DTYPE, device=self.device)
        actions = torch.tensor(sp["action"], dtype=DTYPE, device=self.device).view((-1, self.action_dim))
        rewards = torch.tensor(sp["reward"], dtype=DTYPE, device=self.device).view((-1, 1))
        next_states = torch.tensor(sp["next_state"], dtype=DTYPE, device=self.device)
        dones = torch.tensor(sp["done"], dtype=DTYPE, device=self.device).view((-1, 1))

        # Update the critic network
        ## Calculate the next q values
        ### Don't add noise here !

        ### Calculate the next actions
        with torch.no_grad():
            ### Calculate a_{t+1} to tensor
            next_actions = self.target_actor(next_states)
            # next_actions = torch.tensor(next_actions, dtype=DTYPE, device=DEVICE)

            ### Create the (s_{t+1},a_{t+1}) in the form of tensor
            next_input = torch.cat((next_states, next_actions), dim=-1)

            ### Calculate next_q_values
            next_q_values = self.target_critic(next_input)
            y = rewards + self.gamma * next_q_values * (1 - dones)

        ### Calculate current_q_values
        current_input = torch.cat((states, actions), dim=-1)
        current_q_values = self.critic(current_input)

        ### Calculate the critic loss
        crtitc_loss = self.critic_lossfunc(y, current_q_values).mean()

        ### Update
        self.critic_optimizer.zero_grad()
        crtitc_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        ### Don't add noise here !

        #### Version 1
        # current_now_actions = self.actor(states)
        # current_now_actions = torch.tensor(current_now_actions, dtype=DTYPE, device=DEVICE)
        # current_input = torch.cat((states, current_now_actions), dim=-1)
        # monitor = self.critic(current_input)
        # actor_loss = -1 * torch.mean(monitor)

        #### Version 2
        a = self.actor(states)
        actor_loss = -1 * torch.mean(self.critic(torch.cat((states, a), dim=-1))) + 5 * a.pow(2).mean()

        ### Update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return actor_loss.data.cpu(), crtitc_loss.data.cpu()
