"""
A dict recording the important hyperparameters.
"""
from base_declaration import *
import os

process_ppo_config = {
    "algo": "Distributed PPO",
    "actor_lr": 0.0001,
    "critic_lr": 0.001,
    "num_episode": 1600,
    "num_iteration": 10,
    "gamma": 0.9,
    "lmbda": 0.9,
    "hidden_dim": 128,
    "epoch": 10,
    "epsilon": 0.2,
    "seed": 2023,
    "num_workers": 6,
    "env_name": 'InvertedPendulum-v2',
    "model_dir": './ckpts/dppo/',
    "result_dir": './results/dppo/',
    "device": DEVICE1,
}
