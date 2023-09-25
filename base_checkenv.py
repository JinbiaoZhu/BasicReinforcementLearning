import gym
from gym import envs
# 更新到0.26.2之后更换为 env_list = envs.registry.keys()
env_list = envs.registry.all()
env_ids = [env_item.id for env_item in env_list]
print('There are {0} envs in gym'.format(len(env_ids)))
print(env_ids)