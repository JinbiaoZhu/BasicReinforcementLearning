"""
A dict recording the important hyperparameters.
"""
from base_declaration import *
ddpg_config = {
    "algo": "DDPG",
    "actor_lr": 0.0001,
    "critic_lr": 0.001,
    "num_episode": 10000,
    "num_iteration": 10,
    "target_frequency": 10,
    "gamma": 0.98,
    "tau": 0.005,
    "hidden_dim": 128,
    "minimal_size": 5000,
    "batch_size": 4096,
    "sigma": 0.01,
    "seed": 1,
    "maxlen": 200000,
    "env_name": 'Hopper-v2',
    "model_dir": './ckpts/ddpg/',
    "result_dir": './results/ddpg/',
    "device": DEVICE1
}
