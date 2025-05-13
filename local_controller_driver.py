import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random

from dual_stage_model import LocalController, ControllerQNetwork
from local_controller_runner import RLRunner
from parameter import *
import torch.nn.functional as F

writer = SummaryWriter(train_path)

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 主函数
def main():
    # 设置随机种子
    set_seed(42)
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_device = torch.device('cpu')
    # 初始化Ray
    ray.init(num_cpus=NUM_META_AGENT, num_gpus=1)
    # 创建全局网络
    actor = LocalController(state_dim=4, action_dim=2).to(device)
    actor_target = LocalController(state_dim=4, action_dim=2).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    
    critic = ControllerQNetwork(state_dim=4, action_dim=2).to(device)
    critic_target = ControllerQNetwork(state_dim=4, action_dim=2).to(device)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)
    
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    # 创建DataParallel包装器
    dp_actor = nn.DataParallel(actor)
    dp_critic = nn.DataParallel(critic)
    dp_actor_target = nn.DataParallel(actor_target)
    dp_critic_target = nn.DataParallel(critic_target)
    
    # 启动元代理
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]
    
    # 获取全局网络权重
    weights_set = []
    if device != local_device:
        actor_weights = actor.to(local_device).state_dict()
        actor.to(device)
    else:
        actor_weights = actor.to(local_device).state_dict()
    weights_set.append(actor_weights)

    # 启动第一个任务
    curr_episode = 0
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))

    # 初始化指标收集器
    metric_name = ['travel_dist', 'success_rate', 'explored_rate', 'collision_count']
    training_data = []
    perf_metrics = {n: [] for n in metric_name}

    # 初始化训练回放缓冲区
    experience_buffer = [[] for _ in range(5)]

    # 更新计数器
    update_counter = 0

    # 收集数据并进行训练
    print("Starting training")
    try:
        while True:
            # 等待任何作业完成
            done_id, job_list = ray.wait(job_list)
            # 获取结果
            done_jobs = ray.get(done_id)
            
            # 保存经验和指标
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    # print(len(experience_buffer[i]), i)
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])
            
            # 启动新任务
            curr_episode += 1
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
                    update_counter += 1
                    
                    # 随机采样批次数据
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])
                    
                    # 准备状态和动作数据
                    current_state = torch.stack(rollouts[0]).to(device)  # 
                    next_state = torch.stack(rollouts[1]).to(device)
                    action = torch.stack(rollouts[2]).to(device)
                    reward = torch.stack(rollouts[3]).to(device)
                    done = torch.stack(rollouts[4]).to(device)
                    
                    # TD3算法实现
                    with torch.no_grad():
                        # 选择下一个动作并添加噪声（目标策略平滑）
                        noise = torch.randn_like(action) * POLICY_NOISE
                        noise = torch.clamp(noise, -NOISE_CLIP, NOISE_CLIP)
                        
                        next_action = dp_actor_target(next_state)
                        next_action = torch.clamp(next_action + noise, -1, 1)
                        
                        # 计算目标Q值
                        target_q1, target_q2 = dp_critic_target(next_state, next_action)
                        target_q = torch.min(target_q1, target_q2)
                        target_q = reward + (1 - done) * GAMMA * target_q
                    
                    # 计算当前Q值
                    current_q1, current_q2 = dp_critic(current_state, action)
                    
                    # Critic损失
                    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
                    
                    # 更新Critic
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                    critic_optimizer.step()
                    
                    # 延迟策略更新
                    if update_counter % POLICY_DELAY == 0:
                        # 计算Actor损失（最大化Q值）
                        q1_value, _ = dp_critic(current_state, dp_actor(current_state))
                        actor_loss = -q1_value.mean()
                        # 更新Actor
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                        actor_optimizer.step()
                        
                        # 软更新目标网络
                        with torch.no_grad():
                            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                            
                            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                
                # 记录训练数据
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                
                # 为TD3调整的训练数据记录
                data = [
                    reward.mean().item(),  # 平均奖励
                    target_q.mean().item(),  # 目标Q值
                    actor_loss.item() if update_counter % POLICY_DELAY == 0 else 0,  # Actor损失
                    critic_loss.item(),  # Critic损失
                    0,  # TD3没有熵
                    actor_grad_norm.item() if update_counter % POLICY_DELAY == 0 else 0,  # Actor梯度范数
                    critic_grad_norm.item(),  # Critic梯度范数
                    0,  # TD3没有alpha
                    0,  # TD3没有alpha损失
                    *perf_data  # 性能指标
                ]
                
                training_data.append(data)
                
                # 写入TensorBoard
                if len(training_data) >= SUMMARY_WINDOW:
                    write_to_tensor_board(writer, training_data, curr_episode)
                    training_data = []
                    perf_metrics = {n: [] for n in metric_name}

                # 保存模型
                if curr_episode % SAVE_INTERVAL == 0:
                    print(f"Saving model at episode {curr_episode}")
                    checkpoint = {
                        'actor_model': actor.state_dict(),
                        'critic_model': critic.state_dict(),
                        'actor_target_model': actor_target.state_dict(),
                        'critic_target_model': critic_target.state_dict(),
                        'actor_optimizer': actor_optimizer.state_dict(),
                        'critic_optimizer': critic_optimizer.state_dict(),
                        'episode': curr_episode
                    }
                    torch.save(checkpoint, f"{model_path}/checkpoint.pth")

    except KeyboardInterrupt:
        print("Training interrupted by user")

    finally:
        # 保存最终模型
        print("Saving final model")
        checkpoint = {
            'actor_model': actor.state_dict(),
            'critic_model': critic.state_dict(),
            'actor_target_model': actor_target.state_dict(),
            'critic_target_model': critic_target.state_dict(),
            'actor_optimizer': actor_optimizer.state_dict(),
            'critic_optimizer': critic_optimizer.state_dict(),
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

