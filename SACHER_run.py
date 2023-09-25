"""
Run SAC+HER.
"""

import os

import gym
import numpy as np
from tqdm import tqdm

from SACHER_config import sacher_config
from SACHER_algo import SACHER
from evaluation import SimpleEvaluate
from tools import set_seed, print_dict

if __name__ == '__main__':

    print_dict(os.path.basename(__file__), sacher_config)

    env = gym.make(sacher_config["env_name"])

    num_episodes = sacher_config["num_episode"]
    num_iteration = sacher_config["num_iteration"]

    # state_dim actually is the observation_dim
    state_dim = env.observation_space["observation"].shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    max_action = env.action_space.high[0]

    set_seed(env, sacher_config["seed"])

    agent = SACHER(state_dim, action_dim,
                   hidden_dim=sacher_config["hidden_dim"],
                   max_action=max_action,
                   env=env,
                   goal_dim=goal_dim,
                   activate="relu",
                   actor_lr=sacher_config["actor_lr"],
                   critic_lr=sacher_config["critic_lr"],
                   alpha_lr=sacher_config["alpha_lr"],
                   target_entropy=-1 * action_dim,
                   gamma=sacher_config["gamma"],
                   tau=sacher_config["tau"],
                   batch_size=sacher_config["batch_size"],
                   her_ratio=sacher_config["her_ratio"],
                   maxlen=sacher_config["maxlen"],
                   device=sacher_config["device"],
                   )

    evaluation = SimpleEvaluate(sacher_config["result_dir"],
                                algo_name=sacher_config["algo"], env_name=sacher_config["env_name"],
                                requires_loss=True, save_results=True)

    for i in range(num_iteration):
        with tqdm(total=int(num_episodes / num_iteration), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / num_iteration)):

                evaluation.episode_return_is_zero()

                state = env.reset()
                done = False

                while not done:
                    action, _ = agent.action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.store(state, action, reward, next_state, done)
                    state = next_state
                    env.render()

                    if agent.replay_buffer.__len__() > sacher_config["minimal_size"]:
                        for _ in range(1):
                            lv = agent.update()
                            evaluation.add_single_update_loss(lv)

                    evaluation.add_single_step_reward(reward)

                agent.her()

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
