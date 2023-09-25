"""
Run DDPG.
"""

import os

import gym
import numpy as np
from tqdm import tqdm

from DDPG_algo import DDPG
from DDPG_config import ddpg_config
from evaluation import SimpleEvaluate
from tools import set_seed, print_dict

if __name__ == '__main__':

    print_dict(os.path.basename(__file__), ddpg_config)

    env = gym.make(ddpg_config["env_name"])

    num_episodes = ddpg_config["num_episode"]
    num_iteration = ddpg_config["num_iteration"]

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    set_seed(env, ddpg_config["seed"])

    agent = DDPG(state_dim, action_dim,
                 hidden_dim=ddpg_config["hidden_dim"],
                 max_action=max_action,
                 activate="relu",
                 sigma=ddpg_config["sigma"],
                 actor_lr=ddpg_config["actor_lr"],
                 critic_lr=ddpg_config["critic_lr"],
                 gamma=ddpg_config["gamma"],
                 tau=ddpg_config["tau"],
                 batch_size=ddpg_config["batch_size"],
                 maxlen=ddpg_config["maxlen"],
                 device=ddpg_config["device"],
                 )

    evaluation = SimpleEvaluate(ddpg_config["result_dir"],
                                algo_name=ddpg_config["algo"], env_name=ddpg_config["env_name"],
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
                    env.render()
                    evaluation.add_single_step_reward(reward)

                    if agent.replay_buffer.size() > ddpg_config["minimal_size"]:
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
