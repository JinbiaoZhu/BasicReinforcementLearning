"""
Run DQN.
"""
import os

import gym
import numpy as np
from tqdm import tqdm

from DQN_algo import DQN
from DQN_config import dqn_config
from evaluation import SimpleEvaluate
from tools import set_seed, print_dict

if __name__ == '__main__':

    print_dict(os.path.basename(__file__), dqn_config)

    env = gym.make(dqn_config["env_name"])

    num_episodes = dqn_config["num_episode"]
    num_iteration = dqn_config["num_iteration"]

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    set_seed(env, dqn_config["seed"])

    agent = DQN(state_dim, action_dim,
                hidden_dim=dqn_config["hidden_dim"],
                lr=dqn_config["lr"],
                gamma=dqn_config["gamma"],
                epsilon=dqn_config["epsilon"],
                batch_size=dqn_config["batch_size"],
                target_f=dqn_config["target_frequency"])

    evaluation = SimpleEvaluate(dqn_config["result_dir"],
                                algo_name=dqn_config["algo"], env_name=dqn_config["env_name"],
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

                    if agent.replay_buffer.size() > dqn_config["minimal_size"]:
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
