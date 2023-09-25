"""
Run REINFORCE.
"""
import os

import gym
import numpy as np
from tqdm import tqdm

from REINFORCE_algo import REINFORCE
from REINFORCE_config import reinforce_config
from evaluation import SimpleEvaluate
from tools import set_seed, print_dict

if __name__ == '__main__':

    print_dict(os.path.basename(__file__), reinforce_config)

    env = gym.make(reinforce_config["env_name"])

    num_episodes = reinforce_config["num_episode"]
    num_iteration = reinforce_config["num_iteration"]

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    set_seed(env, reinforce_config["seed"])

    agent = REINFORCE(state_dim, action_dim,
                      hidden_dim=reinforce_config["hidden_dim"],
                      max_action=max_action,
                      lr=reinforce_config["lr"],
                      gamma=reinforce_config["gamma"],
                      batch_size=reinforce_config["batch_size"])

    evaluation = SimpleEvaluate(reinforce_config["result_dir"],
                                algo_name=reinforce_config["algo"], env_name=reinforce_config["env_name"],
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
