"""
The implementation of multi processing.
"""
import copy

import torch
from torch.multiprocessing import set_start_method
from tools import moving_average
import matplotlib.pyplot as plt
import os
from tools import time_string


class MultiProcessing:
    def __init__(self, agent, env, buffer, num_workers, seed, device):

        # Pass parameters
        self.agent = agent
        self.num_workers = num_workers
        self.device = device

        # Create lists
        # Offer one env to each process
        self.env_list = [env for _ in range(self.num_workers)]
        # Offer buffer env to each process
        self.buffer_list = [buffer for _ in range(self.num_workers)]

        # Initialize a list to capture the results.
        self.result_list = []
        self.evalua_list = []

        # Seed
        for e in self.env_list:
            e.seed(seed)

        # Set with "spawn" in Linux
        set_start_method("spawn")

    def obtain_rollouts(self, call_func):

        # Before get results, clear the self.result_list
        self.result_list.clear()

        # Create multiprocessing
        mp = torch.multiprocessing.Pool(self.num_workers)

        # Use cpu for faster
        self.agent.to("cpu")

        # Use temp list to record the rewards
        temp_rewards = []

        for i in range(self.num_workers):
            # res = mp.apply_async(rollout, (self.env_list[i], self.agent.policy, self.buffer_list[i]))
            res = mp.apply_async(call_func, (self.env_list[i], self.agent.policy, self.buffer_list[i]))
            result = res.get()
            self.result_list.append(result[0])
            temp_rewards.append(result[1])


        mp.close()
        mp.join()
        self.evalua_list.append(sum(temp_rewards) / len(temp_rewards))

        # print("Obtain rollouts successfully!")

    def obtain_grads(self, call_func):

        # Offer agent to each process
        agent_list = [copy.deepcopy(self.agent) for _ in range(self.num_workers)]
        for i in range(self.num_workers):
            agent_list[i].to(self.device)

        # Set data container
        self.grad_list = []

        # Create multiprocessing
        mp = torch.multiprocessing.Pool(self.num_workers)

        for i in range(self.num_workers):
            # print(f"------------{id(self.agent)}")
            res = mp.apply_async(call_func, (agent_list[i], self.result_list[i], self.device))
            result = res.get()
            self.grad_list.append(result)

        mp.close()
        mp.join()

        # print("Obtain grads successfully!")

    def model_update(self, call_func):
        agent = copy.deepcopy(self.agent)
        res_agent = call_func(self.grad_list, agent)
        self.agent = res_agent

    def plot_performance(self, save_path="./results/ppo_dist/"):

        plot_return_list = self.evalua_list
        episodes_list = list(range(len(plot_return_list)))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.figure()
        plt.plot(episodes_list, plot_return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        name = save_path + f'InvertedPendulum-' + time_string() + '.png'
        plt.savefig(name)

        plt.show()
