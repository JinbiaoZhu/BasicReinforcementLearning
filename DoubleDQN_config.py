"""
A dict recording the important hyperparameters.
"""
double_dqn_config = {
    "algo": "DoubleDQN",
    "lr": 0.003,
    "num_episode": 500,
    "num_iteration": 10,
    "target_frequency": 10,
    "gamma": 0.98,
    "hidden_dim": 128,
    "minimal_size": 500,
    "batch_size": 64,
    "epsilon": 0.01,
    "seed": 0,
    "env_name": 'CartPole-v0',
    "model_dir": './ckpts/doubledqn/',
    "result_dir": './results/doubledqn/',
}
