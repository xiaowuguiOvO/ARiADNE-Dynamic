import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from local_planner_env import LocalPlannerEnv
from dual_stage_agent import DualStageAgent
from test_local_controller import test_local_controller
from parameter import *

def train_local_controller(
    num_episodes=1000,
    max_steps_per_episode=200,
    lr=3e-4,
    gamma=0.99,
    batch_size=64,
    buffer_size=100000,
    update_every=4,
    tau=1e-3,
    start_epsilon=1.0,
    end_epsilon=0.01,
    epsilon_decay=0.995,
    test_interval=100,
    save_interval=100,
    render_training=False,
    model_path=None,
    save_dir='./models'
):
    """
    在LocalPlannerEnv环境中训练DualStageAgent的localcontroller
    
    参数:
        num_episodes: 训练回合数
        max_steps_per_episode: 每回合最大步数
        lr: 学习率
        gamma: 折扣因子
        batch_size: 批量大小
        buffer_size: 经验回放缓冲区大小
        update_every: 每隔多少步更新一次网络
        tau: 软更新参数
        start_epsilon: 初始探索率
        end_epsilon: 最终探索率
        epsilon_decay: 探索率衰减因子
        test_interval: 测试间隔(每多少回合测试一次)
        save_interval: 保存间隔(每多少回合保存一次模型)
        render_training: 是否渲染训练过程
        model_path: 加载预训练模型路径
        save_dir: 模型保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建环境
    render_mode = 'human' if render_training else None
    env = LocalPlannerEnv(
        map_size=20.0,
        target_radius=0.5,
        max_steps=max_steps_per_episode,
        render_mode=render_mode
    )
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    
    # 创建agent和优化器
    agent = DualStageAgent(device=device)
    agent.env = env  # 添加环境引用
    
    # 创建经验回放缓冲区
    class ReplayBuffer:
        def __init__(self, capacity):
            self.capacity = capacity
            self.buffer = []
            self.position = 0
            
        def push(self, state, action, reward, next_state, done):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity
            
        def sample(self, batch_size):
            batch = np.random.choice(len(self.buffer), batch_size, replace=False)
            states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
            return states, actions, rewards, next_states, dones
        
        def __len__(self):
            return len(self.buffer)
    
    # 创建回放缓冲区
    replay_buffer = ReplayBuffer(buffer_size)
    
    # 创建优化器
    optimizer = optim.Adam(agent.local_controller.parameters(), lr=lr)
    
    # 加载预训练模型(如果提供)
    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            if 'local_controller' in state_dict:
                agent.local_controller.load_state_dict(state_dict['local_controller'])
                print(f"成功加载localcontroller模型: {model_path}")
            else:
                agent.local_controller.load_state_dict(state_dict)
                print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
    
    # 训练跟踪
    rewards_history = []
    epsilon = start_epsilon
    best_reward = -float('inf')
    step_count = 0
    
    print("开始训练...")
    for episode in range(1, num_episodes+1):
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # 当前回合信息
        start_time = time.time()
        
        for step in range(max_steps_per_episode):
            # 获取状态
            robot_state = torch.FloatTensor(obs['robot_state']).unsqueeze(0).to(device)
            target_position = torch.FloatTensor(obs['target_position']).unsqueeze(0).to(device)
            
            # epsilon-贪婪策略选择动作
            if np.random.random() < epsilon:
                # 探索: 随机动作
                action = env.action_space.sample()
                velocity = torch.FloatTensor(action).unsqueeze(0).to(device)
            else:
                # 利用: 使用策略
                with torch.no_grad():
                    velocity, _, _ = agent.local_controller(robot_state, target_position)
                action = velocity.cpu().numpy()[0]
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 保存经验到回放缓冲区
            replay_buffer.push(
                (robot_state.cpu().numpy()[0], target_position.cpu().numpy()[0]),
                action,
                reward,
                (torch.FloatTensor(next_obs['robot_state']).cpu().numpy(), 
                 torch.FloatTensor(next_obs['target_position']).cpu().numpy()),
                float(done)
            )
            
            # 更新状态和统计信息
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            step_count += 1
            
            # 训练网络
            if len(replay_buffer) > batch_size and step_count % update_every == 0:
                # 从回放缓冲区采样
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # 准备批次数据
                robot_states_batch = torch.FloatTensor(np.vstack([s[0] for s in states])).to(device)
                target_positions_batch = torch.FloatTensor(np.vstack([s[1] for s in states])).to(device)
                actions_batch = torch.FloatTensor(np.vstack(actions)).to(device)
                rewards_batch = torch.FloatTensor(np.vstack(rewards)).to(device)
                next_robot_states_batch = torch.FloatTensor(np.vstack([ns[0] for ns in next_states])).to(device)
                next_target_positions_batch = torch.FloatTensor(np.vstack([ns[1] for ns in next_states])).to(device)
                dones_batch = torch.FloatTensor(np.vstack(dones)).to(device)
                
                # 计算当前Q值
                velocity, log_probs, means, log_stds = agent.local_controller.get_action_and_logprob(
                    robot_states_batch, target_positions_batch)
                
                # 计算目标值 (简单的策略梯度)
                # 这里我们采用A2C风格的目标函数: log_prob * (reward + gamma * V(next_state) - V(state))
                with torch.no_grad():
                    next_velocity, _, _, _ = agent.local_controller.get_action_and_logprob(
                        next_robot_states_batch, next_target_positions_batch)
                    
                    # 简单的奖励估计
                    estimated_returns = rewards_batch + gamma * (1 - dones_batch) * 0.5
                
                # 策略梯度损失
                policy_loss = -(log_probs * estimated_returns).mean()
                
                # 熵正则化，增加探索
                entropy = 0.5 + 0.5 * np.log(2 * np.pi) + log_stds.mean()
                entropy_loss = -0.01 * entropy
                
                # 总损失
                loss = policy_loss + entropy_loss
                
                # 梯度下降
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.local_controller.parameters(), 1.0)
                optimizer.step()
            
            # 检查是否结束
            if done:
                break
            
            # 控制渲染速度
            if render_training:
                time.sleep(0.01)
        
        # 更新epsilon
        epsilon = max(end_epsilon, epsilon * epsilon_decay)
        
        # 记录奖励
        rewards_history.append(episode_reward)
        
        # 计算每秒步数
        steps_per_second = episode_steps / (time.time() - start_time) if time.time() > start_time else 0
        
        # 打印进度
        print(f"回合 {episode}/{num_episodes} | 步数: {episode_steps} | 奖励: {episode_reward:.2f} | "
              f"Epsilon: {epsilon:.4f} | 步数/秒: {steps_per_second:.1f}")
        
        # 定期测试
        if episode % test_interval == 0:
            print(f"\n===== 测试回合 {episode} =====")
            test_results = test_local_controller(
                model_path=None,  # 直接使用当前模型
                num_episodes=3,
                max_steps=300,
                render=True
            )
            print("===== 测试完成 =====\n")
            
            # 保存最佳模型
            avg_reward = test_results['avg_reward']
            if avg_reward > best_reward:
                best_reward = avg_reward
                model_save_path = os.path.join(save_dir, f'best_local_controller.pth')
                torch.save(agent.local_controller.state_dict(), model_save_path)
                print(f"保存最佳模型，平均奖励: {best_reward:.2f}")
        
        # 定期保存
        if episode % save_interval == 0:
            model_save_path = os.path.join(save_dir, f'local_controller_episode_{episode}.pth')
            torch.save(agent.local_controller.state_dict(), model_save_path)
            print(f"保存模型到 {model_save_path}")
    
    # 训练结束，保存最终模型
    final_model_path = os.path.join(save_dir, 'local_controller_final.pth')
    torch.save(agent.local_controller.state_dict(), final_model_path)
    print(f"训练完成! 最终模型保存到 {final_model_path}")
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('训练奖励曲线')
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_rewards.png'))
    plt.show()
    
    # 关闭环境
    env.close()
    
    return rewards_history

if __name__ == "__main__":
    # 训练参数
    train_local_controller(
        num_episodes=1000,
        max_steps_per_episode=200,
        lr=3e-4,
        gamma=0.99,
        batch_size=64,
        buffer_size=100000,
        update_every=4,
        tau=1e-3,
        start_epsilon=1.0,
        end_epsilon=0.01,
        epsilon_decay=0.995,
        test_interval=100,
        save_interval=100,
        render_training=False,
        model_path=None,
        save_dir='./models'
    ) 