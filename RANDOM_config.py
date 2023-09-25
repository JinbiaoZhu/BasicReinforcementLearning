"""
A dict recording the important hyperparameters.
"""
from base_declaration import *
random_config = {
    "algo": "RANDOM",
    "actor_lr": 0.001,
    "critic_lr": 0.01,
    "num_episode": 1000,
    "num_iteration": 10,
    "gamma": 0.98,
    "tau": 0.005,
    "hidden_dim": 128,
    "minimal_size": 1000,
    "batch_size": 64,
    "sigma": 0.01,
    "seed": 1,
    "maxlen": 10000,
    "env_name": 'Hopper-v2',
    "model_dir": './ckpts/random/',
    "result_dir": './results/random/',
    "device": DEVICE,
}
