"""
A dict recording the important hyperparameters.
"""
from base_declaration import *
ddpgher_config = {
    "algo": "DDPG+HER",
    "actor_lr": 0.0003,
    "critic_lr": 0.003,
    "num_episode": 500,
    "num_iteration": 10,
    "target_frequency": 10,
    "gamma": 0.98,
    "tau": 0.005,
    "hidden_dim": 128,
    "minimal_size": 1000,
    "batch_size": 512,
    "her_ratio": 0.8,
    "sigma": 0.8,
    "seed": 1,
    "maxlen": 10000,
    "env_name": 'FetchReach-v1',
    "model_dir": './ckpts/ddpg_her/',
    "result_dir": './results/ddpg_her/',
    "device": DEVICE1
}
