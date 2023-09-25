"""
A dict recording the important hyperparameters.
"""
from base_declaration import *
trpo_config = {
    "algo": "TRPO",
    "critic_lr": 0.01,
    "kl_constraint": 0.00005,
    "alpha": 0.5,
    "num_episode": 2000,
    "num_iteration": 10,
    "gamma": 0.9,
    "lmbda": 0.9,
    "hidden_dim": 128,
    "batch_size": 64,
    "seed": 1,
    "env_name": 'Pendulum-v1',
    "model_dir": './ckpts/trpo/',
    "result_dir": './results/trpo/',
    "device": DEVICE1,
}