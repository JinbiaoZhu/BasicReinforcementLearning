"""
Run DPPO.
"""
import gym
import torch.multiprocessing as mp
import torch
from tqdm import tqdm
import numpy as np
from DPPO_algo import child_func
from DPPO_config import process_ppo_config
from continuous_policy import ContinuousPolicyNormal
from continuous_critic import ContinuousValueCritic
import random
from evaluation import SimpleEvaluate
from tools import print_dict

if __name__ == "__main__":

    """
    1. 模型初始化在gpu;
    2. 每个线程针对同一个这样的模型做前向，收集rollout;
    3. 对这个rollout计算损失,对模型参数求梯度;
    4. 最后返回这些梯度再整合。
    """

    print_dict("dict", process_ppo_config)

    mp.set_start_method("spawn", force=True)

    num_episode = process_ppo_config["num_episode"]
    num_iteration = process_ppo_config["num_iteration"]

    global_env = gym.make(process_ppo_config["env_name"])
    global_env.seed(random.randint(0, process_ppo_config["seed"]))

    state_dim = global_env.observation_space.shape[0]
    action_dim = global_env.action_space.shape[0]
    max_action = global_env.action_space.high[0]

    process_ppo_config["state_dim"], process_ppo_config["action_dim"], process_ppo_config[
        "max_action"] = state_dim, action_dim, max_action

    global_nets = [ContinuousPolicyNormal(state_dim, action_dim, max_action, process_ppo_config["hidden_dim"], "relu",
                                          device=process_ppo_config["device"]).to(device=process_ppo_config["device"]),
                   ContinuousValueCritic(state_dim, action_dim, max_action, process_ppo_config["hidden_dim"],
                                         "relu").to(device=process_ppo_config["device"])]

    num_workers = process_ppo_config["num_workers"]
    pipe_dict = dict((i, (p1, p2)) for i in range(num_workers) for p1, p2 in (mp.Pipe(duplex=True),))

    manager = mp.Manager()  # 创建对象共享池
    shared_list = manager.list()  # 创建共享列表
    queue = mp.Queue(maxsize=num_workers)  # 创建队列
    lock = mp.Lock()  # 创建锁

    child_processes = []
    reward_list = []
    net_list = []

    evaluation = SimpleEvaluate(process_ppo_config["result_dir"],
                                algo_name=process_ppo_config["algo"], env_name=process_ppo_config["env_name"],
                                requires_loss=True, save_results=True)

    for i in tqdm(range(num_workers)):
        pro = mp.Process(target=child_func,
                         args=(pipe_dict[i][1], process_ppo_config, queue, shared_list, global_env))
        child_processes.append(pro)
    [p.start() for p in child_processes]

    for i in range(num_iteration):
        with tqdm(total=int(num_episode / num_iteration), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episode / num_iteration)):

                [pipe_dict[i][0].send(global_nets) for i in range(num_workers)]
                net_list.clear()

                evaluation.episode_return_is_zero()

                for _ in range(num_workers):
                    t_ref = queue.get()
                    t = t_ref[0]
                    net_list.append(t)

                act_model_dict = global_nets[0].state_dict()
                cri_model_dict = global_nets[1].state_dict()

                for k1, k2 in zip(act_model_dict.keys(), cri_model_dict.keys()):
                    result1 = torch.zeros_like(act_model_dict[k1]).to("cpu")
                    result2 = torch.zeros_like(cri_model_dict[k2]).to("cpu")
                    for j in range(num_workers):
                        result1 += net_list[j][0].state_dict()[k1]
                        result2 += net_list[j][1].state_dict()[k2]
                    result1 /= num_workers
                    result2 /= num_workers
                    act_model_dict[k1] = result1.to(device=process_ppo_config["device"])
                    cri_model_dict[k2] = result2.to(device=process_ppo_config["device"])
                global_nets[0].load_state_dict(act_model_dict)
                global_nets[1].load_state_dict(cri_model_dict)

                for index in range(num_workers):
                    evaluation.episode_return += net_list[index][2]
                    evaluation.add_single_update_loss(loss=(net_list[index][3:5]))
                evaluation.episode_return /= num_workers
                evaluation.episode_return_record()

                shared_list[:] = []

                if (i_episode + 1) % num_iteration == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episode / num_iteration * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(evaluation.return_list[-10:])
                    })
                pbar.update(1)

    [p.terminate() for p in child_processes]
    evaluation.plot_performance()
