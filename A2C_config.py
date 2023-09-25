"""
A dict recording the important hyperparameters.
"""
from base_declaration import *
a2c_config = {
    "algo": "A2C",
    "actor_lr": 0.001,
    "critic_lr": 0.01,
    "num_episode": 5000,
    "num_iteration": 10,
    "gamma": 0.995,
    "tau": 0.005,
    "hidden_dim": 128,
    "minimal_size": 1000,
    "batch_size": 64,
    "sigma": 0.01,
    "seed": 123,
    "maxlen": 10000,
    "env_name": 'InvertedPendulum-v2',
    "model_dir": './ckpts/a2c/',
    "result_dir": './results/a2c/',
    "device": DEVICE,
}
