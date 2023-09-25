"""
A dict recording the important hyperparameters.
"""
from base_declaration import *
sac_config = {
    "algo": "SAC",
    "actor_lr": 0.0003,
    "critic_lr": 0.003,
    "alpha_lr": 0.0003,
    "num_episode": 2000,
    "num_iteration": 10,
    "gamma": 0.99,
    "tau": 0.005,
    "hidden_dim": 128,
    "minimal_size": 5000,
    "batch_size": 4096,
    "seed": 1,
    "maxlen": 200000,
    "env_name": 'HalfCheetah-v3',
    "model_dir": './ckpts/sac/',
    "result_dir": './results/sac/',
    "device": DEVICE
}
