"""
Run RANDOM.
"""

import os

import gym
import numpy as np
from tqdm import tqdm

from RANDOM_algo import RANDOM
from RANDOM_config import random_config
from evaluation import SimpleEvaluate
from tools import set_seed, print_dict

if __name__ == '__main__':

    print_dict(os.path.basename(__file__), random_config)

    env = gym.make(random_config["env_name"])

    num_episodes = random_config["num_episode"]
    num_iteration = random_config["num_iteration"]

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    set_seed(env, random_config["seed"])

    agent = RANDOM(state_dim, action_dim,
                   hidden_dim=random_config["hidden_dim"],
                   max_action=max_action,
                   activate="relu",
                   actor_lr=random_config["actor_lr"],
                   critic_lr=random_config["critic_lr"],
                   gamma=random_config["gamma"],
                   batch_size=random_config["batch_size"],
                   maxlen=random_config["maxlen"],
                   device=random_config["device"],
                   )

    evaluation = SimpleEvaluate(random_config["result_dir"],
                                algo_name=random_config["algo"], env_name=random_config["env_name"],
                                requires_loss=True, save_results=True)

    for i in range(num_iteration):
        with tqdm(total=int(num_episodes / num_iteration), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / num_iteration)):

                evaluation.episode_return_is_zero()

                state = env.reset()
                done = False

                while not done:
                    action, dist = agent.action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.store(state, action, reward, next_state, done, dist)
                    state = next_state
                    env.render()
                    evaluation.add_single_step_reward(reward)

                evaluation.episode_return_record()

                lv = agent.update()
                evaluation.add_single_update_loss(lv)

                if (i_episode + 1) % num_iteration == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / num_iteration * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(evaluation.return_list[-10:])
                    })
                pbar.update(1)

    evaluation.plot_performance()
