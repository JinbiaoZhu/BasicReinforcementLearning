"""
A dict recording the important hyperparameters.
"""
from base_declaration import *
ac_config = {
    "algo": "AC",
    "actor_lr": 0.0003,
    "critic_lr": 0.003,
    "num_episode": 5000,
    "num_iteration": 10,
    "gamma": 0.995,
    "tau": 0.005,
    "hidden_dim": 128,
    "minimal_size": 1000,
    "batch_size": 64,
    "sigma": 0.01,
    "seed": 1,
    "maxlen": 10000,
    "env_name": 'Hopper-v2',
    "model_dir": './ckpts/ac/',
    "result_dir": './results/ac/',
    "device": DEVICE,
}
