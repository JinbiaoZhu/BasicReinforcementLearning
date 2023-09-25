"""
A dict recording the important hyperparameters.
"""
dqn_config = {
    "algo": "DQN",
    "lr": 0.003,
    "num_episode": 500,
    "num_iteration": 10,
    "target_frequency": 10,
    "gamma": 0.98,
    "hidden_dim": 64,
    "minimal_size": 500,
    "batch_size": 64,
    "epsilon": 0.01,
    "seed": 400,
    "env_name": 'CartPole-v0',
    "model_dir": './ckpts/dqn/',
    "result_dir": './results/dqn/',
}
