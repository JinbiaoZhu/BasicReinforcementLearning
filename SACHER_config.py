"""
A dict recording the important hyperparameters.
"""
from base_declaration import *

sacher_config = {
    "algo": "SAC+HER",
    "actor_lr": 0.0001,
    "critic_lr": 0.001,
    "alpha_lr": 0.0001,
    "num_episode": 20,
    "num_iteration": 10,
    "gamma": 0.98,
    "tau": 0.005,
    "her_ratio": 0.99,
    "hidden_dim": 128,
    "minimal_size": 5000,
    "batch_size": 4096,
    "seed": 1,
    "maxlen": 100000,
    "env_name": 'FetchPush-v1',
    "model_dir": './ckpts/sac_her/',
    "result_dir": './results/sac_her/',
    "device": DEVICE
}
