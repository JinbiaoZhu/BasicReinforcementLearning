"""
Run DoubleDQN.
"""
import os

import gym
import numpy as np
from tqdm import tqdm

from DoubleDQN_algo import DoubleDQN
from DoubleDQN_config import double_dqn_config
from evaluation import SimpleEvaluate
from tools import set_seed, print_dict

if __name__ == '__main__':

    print_dict(os.path.basename(__file__), double_dqn_config)

    env = gym.make(double_dqn_config["env_name"])

    num_episodes = double_dqn_config["num_episode"]
    num_iteration = double_dqn_config["num_iteration"]

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    set_seed(env, double_dqn_config["seed"])

    agent = DoubleDQN(state_dim, action_dim,
                      hidden_dim=double_dqn_config["hidden_dim"],
                      lr=double_dqn_config["lr"],
                      gamma=double_dqn_config["gamma"],
                      epsilon=double_dqn_config["epsilon"],
                      batch_size=double_dqn_config["batch_size"],
                      target_f=double_dqn_config["target_frequency"])

    evaluation = SimpleEvaluate(double_dqn_config["result_dir"],
                                algo_name=double_dqn_config["algo"], env_name=double_dqn_config["env_name"],
                                requires_loss=True, save_results=True)

    for i in range(num_iteration):
        with tqdm(total=int(num_episodes / num_iteration), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / num_iteration)):

                evaluation.episode_return_is_zero()

                state = env.reset()
                done = False

                while not done:
                    action = agent.action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.store(state, action, reward, next_state, done)
                    state = next_state
                    evaluation.add_single_step_reward(reward)

                    if agent.replay_buffer.size() > double_dqn_config["minimal_size"]:
                        lv = agent.update()
                        evaluation.add_single_update_loss(lv)

                evaluation.episode_return_record()

                if (i_episode + 1) % num_iteration == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / num_iteration * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(evaluation.return_list[-10:])
                    })
                pbar.update(1)

    evaluation.plot_performance()
