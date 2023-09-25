"""
Run SAC.
"""

import os

import gym
import numpy as np
from tqdm import tqdm

from SAC_algo import SAC
from SAC_config import sac_config
from evaluation import SimpleEvaluate
from tools import set_seed, print_dict, render_interval

if __name__ == '__main__':

    print_dict(os.path.basename(__file__), sac_config)

    env = gym.make(sac_config["env_name"])

    num_episodes = sac_config["num_episode"]
    num_iteration = sac_config["num_iteration"]

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    set_seed(env, sac_config["seed"])

    agent = SAC(state_dim, action_dim,
                target_entropy=-1*action_dim,
                hidden_dim=sac_config["hidden_dim"],
                max_action=max_action,
                activate="relu",
                actor_lr=sac_config["actor_lr"],
                critic_lr=sac_config["critic_lr"],
                alpha_lr=sac_config["alpha_lr"],
                gamma=sac_config["gamma"],
                tau=sac_config["tau"],
                batch_size=sac_config["batch_size"],
                maxlen=sac_config["maxlen"],
                device=sac_config["device"],
                )

    evaluation = SimpleEvaluate(sac_config["result_dir"],
                                algo_name=sac_config["algo"], env_name=sac_config["env_name"],
                                requires_loss=True, save_results=True)

    for i in range(num_iteration):
        with tqdm(total=int(num_episodes / num_iteration), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / num_iteration)):

                evaluation.episode_return_is_zero()

                state = env.reset()
                done = False

                while not done:
                    action, _ = agent.action(state)
                    next_state, reward, done, _ = env.step(action.data.cpu().numpy())
                    agent.store(state, action.data.cpu().numpy(), reward, next_state, done)
                    state = next_state
                    render_interval(i_episode, 1, env)
                    evaluation.add_single_step_reward(reward)

                    if agent.replay_buffer.size() > sac_config["minimal_size"]:
                        for __ in range(1):
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
