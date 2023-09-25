"""
Run DDPG+HER.
"""

import os

import gym
import numpy as np
from tqdm import tqdm

from HER_config import ddpgher_config
from HER_algo import DDPG_HER
from evaluation import SimpleEvaluate
from tools import set_seed, print_dict

if __name__ == '__main__':

    print_dict(os.path.basename(__file__), ddpgher_config)

    env = gym.make(ddpgher_config["env_name"])

    num_episodes = ddpgher_config["num_episode"]
    num_iteration = ddpgher_config["num_iteration"]

    # state_dim actually is the observation_dim
    state_dim = env.observation_space["observation"].shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    max_action = env.action_space.high[0]

    set_seed(env, ddpgher_config["seed"])

    agent = DDPG_HER(state_dim, action_dim, hidden_dim=ddpgher_config["hidden_dim"],
                     max_action=max_action,
                     goal_dim=goal_dim,
                     env=env,
                     activate="relu",
                     sigma=ddpgher_config["sigma"],
                     actor_lr=ddpgher_config["actor_lr"],
                     critic_lr=ddpgher_config["critic_lr"],
                     gamma=ddpgher_config["gamma"],
                     tau=ddpgher_config["tau"],
                     batch_size=ddpgher_config["batch_size"],
                     her_ratio=ddpgher_config["her_ratio"],
                     maxlen=ddpgher_config["maxlen"],
                     device=ddpgher_config["device"],
                     )

    evaluation = SimpleEvaluate(ddpgher_config["result_dir"],
                                algo_name=ddpgher_config["algo"], env_name=ddpgher_config["env_name"],
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

                    if agent.replay_buffer.__len__() > ddpgher_config["minimal_size"]:
                        for _ in range(1):
                            lv = agent.update()
                            evaluation.add_single_update_loss(lv)

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
