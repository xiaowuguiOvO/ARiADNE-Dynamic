import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
import swanlab
from dual_stage_model import WaypointSelector, WayPointQNet
from runner import RLRunner
from parameter import *
from tqdm import tqdm
writer = SummaryWriter(train_path)

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

MAX_EPISODES = 10000
episode_bar = tqdm(total=MAX_EPISODES, desc="Training Progress")

# 主函数
def main():
    # 设置随机种子
    set_seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_device = torch.device('cpu')
    
    # 初始化Ray
    ray.init(num_cpus=NUM_META_AGENT, num_gpus=1)
    
    # 创建模型
    node_dim = NODE_INPUT_DIM  # 从parameter.py中获取
    embedding_dim = EMBEDDING_DIM  # 从parameter.py中获取
    
    # 创建全局网络
    global_policy_net = WaypointSelector(node_dim, embedding_dim).to(device)
    global_q_net1 = WayPointQNet(node_dim, embedding_dim).to(device)
    global_q_net2 = WayPointQNet(node_dim, embedding_dim).to(device)
    global_target_q_net1 = WayPointQNet(node_dim, embedding_dim).to(device)
    global_target_q_net2 = WayPointQNet(node_dim, embedding_dim).to(device)
    
    # 创建优化器
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=1e-4)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=1e-4)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=1e-4)
    
    # 初始化log_alpha
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))
    # 加载模型（如果需要）
    curr_episode = 0
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=device)
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha = checkpoint['log_alpha']
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
        
        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        curr_episode = checkpoint['episode']
        
        print(f"curr_episode set to {curr_episode}")
    
    # 加载目标网络
    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()
    
    # 创建DataParallel包装器
    dp_policy = nn.DataParallel(global_policy_net)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)
    
    # 启动元代理
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]
    
    # 获取全局网络权重
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        global_policy_net.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
    weights_set.append(policy_weights)
    
    # 启动第一个任务
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))
    
    # 初始化指标收集器
    metric_name = ['travel_dist', 'success_rate', 'explored_rate', 'collision_count']
    training_data = []
    perf_metrics = {n: [] for n in metric_name}
    
    # 初始化训练回放缓冲区
    experience_buffer = [[] for _ in range(15)]
    
    # 目标Q网络更新计数器
    target_q_update_counter = 0
    
    # 收集数据并进行训练
    print("Starting training")
    # swanlab init
    swanlab.init(project='Ariadne-Dynamic')
    
    try:
        while curr_episode < MAX_EPISODES:
            # 等待任何作业完成
            done_id, job_list = ray.wait(job_list)
            # 获取结果
            done_jobs = ray.get(done_id)
            
            # 保存经验和指标
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])
            
            # 启动新任务
            curr_episode += 1
            episode_bar.update(1)
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))
            
            # 开始训练
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training")
                
                # 保持回放缓冲区大小
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]
                
                indices = range(len(experience_buffer[0]))
                
                # 每步训练多次
                for j in range(8):
                    # 随机采样批次数据
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])
                    
                    # 将批次数据堆叠为张量
                    node_inputs = torch.stack(rollouts[0]).to(device)
                    node_padding_mask = torch.stack(rollouts[1]).to(device)
                    edge_mask = torch.stack(rollouts[2]).to(device)
                    current_index = torch.stack(rollouts[3]).to(device)
                    current_edge = torch.stack(rollouts[4]).to(device)
                    edge_padding_mask = torch.stack(rollouts[5]).to(device)
                    action = torch.stack(rollouts[6]).to(device)
                    reward = torch.stack(rollouts[7]).to(device)
                    done = torch.stack(rollouts[8]).to(device)
                    next_node_inputs = torch.stack(rollouts[9]).to(device)
                    next_node_padding_mask = torch.stack(rollouts[10]).to(device)
                    next_edge_mask = torch.stack(rollouts[11]).to(device)
                    next_current_index = torch.stack(rollouts[12]).to(device)
                    next_current_edge = torch.stack(rollouts[13]).to(device)
                    next_edge_padding_mask = torch.stack(rollouts[14]).to(device)
                    

                    # 准备观察和下一个观察
                    observation = [node_inputs, node_padding_mask, edge_mask, current_index, 
                                         current_edge, edge_padding_mask]
                    next_observation = [next_node_inputs, next_node_padding_mask, next_edge_mask,
                                              next_current_index, next_current_edge, next_edge_padding_mask]
                    
                    # SAC算法实现
                    with torch.no_grad():
                        q_values1 = dp_q_net1(*observation)
                        q_values2 = dp_q_net2(*observation)
                        q_values = torch.min(q_values1, q_values2)
                    
                    # 策略网络前向传播                    
                    logp = dp_policy(*observation)
                    policy_loss = torch.sum(
                            (logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach())),
                            dim=1).mean()
                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100,
                                                                      norm_type=2)
                    global_policy_optimizer.step()

                    with torch.no_grad():
                        next_logp = dp_policy(*next_observation)
                        next_q_values1 = dp_target_q_net1(*next_observation)
                        next_q_values2 = dp_target_q_net2(*next_observation)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(
                            next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                            dim=1).unsqueeze(1)
                        target_q_batch = reward + GAMMA * (1 - done) * value_prime

                    mse_loss = nn.MSELoss()
                    # 计算q值
                    q_values1 = dp_q_net1(*observation)
                    # print(q_values1.shape, action.shape)
                    q1 = torch.gather(q_values1, 1, action.long().unsqueeze(1).unsqueeze(2))
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()
                    # 更新q值
                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net1_optimizer.step()

                    # 计算q值
                    q_values2 = dp_q_net2(*observation)
                    q2 = torch.gather(q_values2, 1, action.long().unsqueeze(1).unsqueeze(2))
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()
                    # 更新q值
                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net2_optimizer.step()

                    # 计算熵
                    entropy = (logp * logp.exp()).sum(dim=-1)
                    # 更新熵
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()
                    # 更新熵
                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()
                    
                    target_q_update_counter += 1
                # 记录训练数据
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward.mean().item(), value_prime.mean().item(), policy_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                        alpha_loss.item(), *perf_data]
                training_data.append(data)
                # # write record to tensorboard
                if len(training_data) >= SUMMARY_WINDOW:
                    write_to_tensor_board(writer, training_data, curr_episode)
                    training_data = []
                    perf_metrics = {}
                    for n in metric_name:
                        perf_metrics[n] = []
                episode_bar.set_postfix({
                    'policy_loss': f"{policy_loss.item():.4f}",
                    'q_loss': f"{(q1_loss.item() + q2_loss.item()):.4f}",
                    'reward': f"{reward.mean().item():.2f}",
                    'success': f"{np.nanmean(perf_metrics['success_rate']):.2f}"
                })
                # swanlab record
                swanlab.log({
                    'loss/value': value_prime.mean().item(),
                    'loss/policy_loss': policy_loss.item(),
                    'loss/q_value_loss': (q1_loss + q2_loss).item(),
                    'loss/entropy': entropy.mean().item(),
                    'loss/policy_grad_norm': policy_grad_norm.item(),
                    'loss/q_value_grad_norm': q_grad_norm.item(),
                    'loss/log_alpha': log_alpha.item(),
                    'loss/alpha_loss': alpha_loss.item(),
                    'perf/reward': reward.mean().item(),
                    'perf/travel_dist': np.nanmean(perf_metrics['travel_dist']),
                    'perf/explored_rate': np.nanmean(perf_metrics['explored_rate']),
                    'perf/success_rate': np.nanmean(perf_metrics['success_rate']),
                    'perf/collision_count': np.nanmean(perf_metrics['collision_count']),
                    'episode': curr_episode
                })

                # 保存模型
                if curr_episode % SAVE_INTERVAL == 0:
                    print(f"Saving model at episode {curr_episode}")
                    checkpoint = {
                        'policy_model': global_policy_net.state_dict(),
                        'q_net1_model': global_q_net1.state_dict(),
                        'q_net2_model': global_q_net2.state_dict(),
                        'policy_optimizer': global_policy_optimizer.state_dict(),
                        'q_net1_optimizer': global_q_net1_optimizer.state_dict(),
                        'q_net2_optimizer': global_q_net2_optimizer.state_dict(),
                        'log_alpha': log_alpha,
                        'log_alpha_optimizer': log_alpha_optimizer.state_dict(),
                        'episode': curr_episode
                    }
                    torch.save(checkpoint, f"{model_path}/checkpoint.pth")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        # 保存最终模型
        print("Saving final model")
        checkpoint = {
            'policy_model': global_policy_net.state_dict(),
            'q_net1_model': global_q_net1.state_dict(),
            'q_net2_model': global_q_net2.state_dict(),
            'policy_optimizer': global_policy_optimizer.state_dict(),
            'q_net1_optimizer': global_q_net1_optimizer.state_dict(),
            'q_net2_optimizer': global_q_net2_optimizer.state_dict(),
            'log_alpha': log_alpha,
            'log_alpha_optimizer': log_alpha_optimizer.state_dict(),
            'episode': curr_episode
        }
        torch.save(checkpoint, f"{model_path}/checkpoint_final.pth")
        
        # 关闭Ray
        ray.shutdown()


def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    reward, value, policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss, travel_dist, success_rate, explored_rate, collision_count = tensorboard_data

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Alpha Loss', scalar_value=alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=q_value_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policy_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=q_value_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Log Alpha', scalar_value=log_alpha, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Collision Count', scalar_value=collision_count, global_step=curr_episode) 


if __name__ == "__main__":
    main()

