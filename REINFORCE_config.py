"""
A dict recording the important hyperparameters.
"""
reinforce_config = {
    "algo": "REINFORCE",
    "lr": 0.0005,
    "num_episode": 100,
    "num_iteration": 10,
    "gamma": 0.98,
    "hidden_dim": 128,
    "batch_size": 64,
    "seed": 1,
    "env_name": 'Hopper-v3',
    "model_dir": './ckpts/reinforce/',
    "result_dir": './results/reinforce/',
    "device": "cuda:0",
}
