#!/usr/bin/env python3
# 测试训练好的模型

from dual_stage_model import LocalController, ControllerQNetwork
import torch
import numpy as np
import argparse
import time
import os
from local_planner_env import LocalPlannerEnv
from dual_stage_agent import DualStageAgent
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="测试训练好的本地控制器模型")
    parser.add_argument('--model-path', type=str, default='./saved_models/best_model', 
                        help='模型路径')
    parser.add_argument('--model-name', type=str, default='best', 
                        help='模型名称')
    parser.add_argument('--episodes', type=int, default=10, 
                        help='测试的episode数量')
    parser.add_argument('--max-steps', type=int, default=200, 
                        help='每个episode的最大步数')
    parser.add_argument('--render', action='store_true', default=True, 
                        help='是否渲染环境')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--record', action='store_true', 
                        help='是否记录测试结果')
    parser.add_argument('--force-cpu', action='store_true',
                        help='强制使用CPU，即使有可用的GPU')
    parser.add_argument('--test-env-only', action='store_true',
                        help='只测试环境，不测试模型')
    return parser.parse_args()

def test_model():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化环境
    render_mode = 'human' if args.render else None
    env = LocalPlannerEnv(
        map_size=10.0,
        target_radius=0.3,
        max_steps=args.max_steps,
        render_mode=render_mode,
        non_blocking_render=True
    )
    
    # 加载模型
    state_dim = 4  # [distance_to_target, theta, v_linear, v_angular]
    action_dim = 2  # [linear_velocity, angular_velocity]
    max_action = 1.0
    
    # 初始化Agent
    agent = DualStageAgent(device=device)
    
    # 尝试加载模型
    model_path = args.model_path
    model_name = args.model_name
    
    try:
        # 尝试多种命名格式加载模型
        possible_model_files = [
            f"{model_path}/{model_name}_actor.pth",
            f"{model_path}/actor_{model_name}.pth",
            f"{model_path}/{model_name}.pth",
            f"{model_path}/best_actor.pth",
            f"{model_path}/actor_best.pth",
            f"{model_path}/actor_good.pth",
            f"{model_path}/good_actor.pth"
        ]
        
        model_loaded = False
        for model_file in possible_model_files:
            if os.path.exists(model_file):
                print(f"Attempting to load model: {model_file}")
                # 显式地将模型移到正确的设备上
                model_weights = torch.load(model_file, map_location=device)
                agent.local_controller.load_state_dict(model_weights)
                agent.local_controller = agent.local_controller.to(device)  # 确保整个模型在指定设备上
                print(f"Successfully loaded model: {model_file} to device: {device}")
                
                # 检查模型权重是否有效
                valid_weights = True
                for name, param in agent.local_controller.named_parameters():
                    if torch.isnan(param).any():
                        print(f"Warning: Model parameter {name} contains NaN values")
                        valid_weights = False
                    if torch.isinf(param).any():
                        print(f"Warning: Model parameter {name} contains infinite values")
                        valid_weights = False
                
                if valid_weights:
                    print("Model weight check passed, all parameters are normal")
                    model_loaded = True
                    break
                else:
                    print("Model weight check failed, trying other model files")
                    continue
        
        if not model_loaded:
            print("All attempts failed, unable to load valid model file")
            return
        
        agent.local_controller.eval()  # 设置为评估模式
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 记录结果
    if args.record:
        results_dir = "./test_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 创建结果文件
        results_file = f"{results_dir}/test_results_{time.strftime('%Y%m%d-%H%M%S')}.txt"
        with open(results_file, 'w') as f:
            f.write("Episode,Steps,TotalReward,Success\n")
    
    # 测试统计
    episode_rewards = []
    episode_steps = []
    success_count = 0
    
    for episode in range(args.episodes):
        obs, info = env.reset()
        x, y, theta, v_linear, v_angular, distance_to_target = obs['robot_state']
        agent.update_robot_state(distance_to_target, theta, v_linear, v_angular)
        
        total_reward = 0
        done = False
        steps = 0
        
        print(f"\nStarting Episode {episode+1}/{args.episodes}")
        print(f"Initial position: ({x:.2f}, {y:.2f}), Orientation: {np.degrees(theta):.2f}°")
        print(f"Target position: ({obs['target_position'][0]:.2f}, {obs['target_position'][1]:.2f})")
        print(f"Distance to target: {distance_to_target:.2f}")
        
        # 记录每个episode的动作和状态
        action_history = []
        state_history = []
        
        while not done:
            try:
                # 获取当前状态
                current_state = agent.get_robot_state()
                state_history.append(current_state)
                
                # 打印当前状态用于调试
                if steps % 20 == 0:
                    print(f"Debug - Current state: {current_state}")
                
                # 使用模型预测动作，确保数据在正确的设备上
                current_state_tensor = torch.tensor(current_state, dtype=torch.float32).to(device)
                if current_state_tensor.dim() == 1:
                    current_state_tensor = current_state_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    # 再次确认模型在正确的设备上
                    agent.local_controller = agent.local_controller.to(device)
                    # 执行前向传播
                    action = agent.local_controller(current_state_tensor)
                    # 打印原始动作用于调试
                    if steps % 20 == 0:
                        print(f"Debug - Model output action: {action.cpu().numpy().squeeze()}")
                    action = action.cpu().numpy().squeeze()
                
                # 记录动作
                action_history.append(action)
                
                # 执行动作
                try:
                    # action = [1, 1]
                    obs, reward, terminated, truncated, info, done = env.step(action)
                    
                    # 打印每一步的信息
                    if steps % 5 == 0:
                        print(f"  Step {steps}: Action=[{action[0]:.2f}, {action[1]:.2f}], Reward={reward:.2f}")
                except Exception as step_error:
                    print(f"Environment step call error: {step_error}")
                    print(f"Action values: {action}")
                    # 尝试使用有效的动作继续
                    action = np.clip(action, -1.0, 1.0)  # 确保动作在有效范围内
                    if np.isnan(action).any():  # 检查是否有NaN值
                        action = np.array([0.5, 0.0])  # 使用安全的默认动作
                        print("NaN action detected, using default action instead")
                    try:
                        obs, reward, terminated, truncated, info, done = env.step(action)
                    except:
                        print("Failed even with safe action, ending episode")
                        done = True
                        break
                
                # 更新状态
                try:
                    x, y, theta, v_linear, v_angular, distance_to_target = obs['robot_state']
                    agent.update_robot_state(distance_to_target, theta, v_linear, v_angular)
                except Exception as state_error:
                    print(f"Error updating state: {state_error}")
                    print(f"Observation: {obs}")
                    done = True
                    break
                
                total_reward += reward
                steps += 1
                
                # 每10步打印一次状态
                if steps % 10 == 0:
                    print(f"Step {steps}: Position=({x:.2f}, {y:.2f}), Orientation={np.degrees(theta):.2f}°, "
                          f"Linear velocity={v_linear:.2f}, Angular velocity={v_angular:.2f}, "
                          f"Distance={distance_to_target:.2f}, Reward={reward:.2f}")
                
            except Exception as e:
                print(f"Error during step execution: {e}")
                print(f"Current state: {current_state}")
                try:
                    print(f"Device info - Model: {next(agent.local_controller.parameters()).device}, Input: {current_state_tensor.device}")
                except:
                    print("Unable to get device information")
                done = True
                break
                
            if terminated or truncated or steps >= args.max_steps:
                success = info.get('reached_target', False)
                print(f"Episode {episode+1} ended: {'Success' if success else 'Failed'}")
                print(f"Total steps: {steps}, Total reward: {total_reward:.2f}")
                
                if success:
                    success_count += 1
                
                episode_rewards.append(total_reward)
                episode_steps.append(steps)
                
                if args.record:
                    with open(results_file, 'a') as f:
                        f.write(f"{episode+1},{steps},{total_reward:.2f},{1 if success else 0}\n")
                
                # 可选：绘制这个episode的动作和状态历史
                if args.record:
                    action_history = np.array(action_history)
                    state_history = np.array(state_history)
                    
                    # 创建图表
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # 绘制动作
                    ax1.plot(action_history[:, 0], label='线速度')
                    ax1.plot(action_history[:, 1], label='角速度')
                    ax1.set_title(f'Episode {episode+1} 动作')
                    ax1.set_xlabel('步数')
                    ax1.set_ylabel('动作值')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # 绘制状态
                    ax2.plot(state_history[:, 0], label='距离')
                    ax2.plot(state_history[:, 1], label='朝向差')
                    ax2.set_title(f'Episode {episode+1} 状态')
                    ax2.set_xlabel('步数')
                    ax2.set_ylabel('状态值')
                    ax2.legend()
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(f"{results_dir}/episode_{episode+1}_plot.png")
                    plt.close()
                
                break
    
    # 打印测试统计
    print("\n===== Test Statistics =====")
    print(f"Number of test episodes: {args.episodes}")
    print(f"Success rate: {success_count}/{args.episodes} = {success_count/args.episodes*100:.2f}%")
    
    # 添加空数组检查
    if len(episode_steps) > 0:
        print(f"Average steps: {np.mean(episode_steps):.2f}")
        print(f"Average reward: {np.mean(episode_rewards):.2f}")
        print(f"Highest reward: {np.max(episode_rewards):.2f}")
    else:
        print("Warning: No episodes completed, cannot calculate statistics")
    
    # 关闭环境
    env.close()

def test_env_only(max_steps=200, render=True):
    """
    仅测试环境功能，使用固定动作
    """
    print("\n===== Testing Environment Functions =====")
    # 初始化环境
    render_mode = 'human' if render else None
    env = LocalPlannerEnv(
        map_size=10.0,
        target_radius=0.3,
        max_steps=max_steps,
        render_mode=render_mode,
        non_blocking_render=True
    )
    
    # 重置环境
    obs, info = env.reset()
    print(f"Initial observation: {obs['robot_state']}")
    print(f"Target position: {obs['target_position']}")
    
    # 使用固定动作进行测试
    total_reward = 0
    for step in range(max_steps):
        # 使用简单的前进动作
        action = np.array([0.5, 0.1])  # 低速前进，轻微转向
        
        # 执行动作
        obs, reward, terminated, truncated, info, done = env.step(action)
        total_reward += reward
        
        # 打印信息
        if step % 10 == 0:
            x, y, theta, v_linear, v_angular, distance_to_target = obs['robot_state']
            print(f"步骤 {step}: 位置=({x:.2f}, {y:.2f}), 距离={distance_to_target:.2f}, 奖励={reward:.2f}")
        
        # 检查是否结束
        if terminated or truncated:
            print(f"Environment test ended at step {step}")
            success = info.get('reached_target', False)
            print(f"{'Successfully reached target' if success else 'Failed to reach target'}")
            break
    
    print(f"Total environment test reward: {total_reward:.2f}")
    env.close()
    print("Environment test completed")
    return True

if __name__ == "__main__":
    # 处理命令行参数
    parser = argparse.ArgumentParser(description="测试训练好的本地控制器模型")
    parser.add_argument('--model-path', type=str, default='./saved_models/best_model', 
                        help='模型路径')
    parser.add_argument('--model-name', type=str, default='best', 
                        help='模型名称')
    parser.add_argument('--episodes', type=int, default=10, 
                        help='测试的episode数量')
    parser.add_argument('--max-steps', type=int, default=200, 
                        help='每个episode的最大步数')
    parser.add_argument('--render', action='store_true', default=True, 
                        help='是否渲染环境')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--record', action='store_true', 
                        help='是否记录测试结果')
    parser.add_argument('--force-cpu', action='store_true',
                        help='强制使用CPU，即使有可用的GPU')
    parser.add_argument('--test-env-only', action='store_true',
                        help='只测试环境，不测试模型')
    args = parser.parse_args()
    
    if args.test_env_only:
        test_env_only(max_steps=args.max_steps, render=args.render)
    else:
        # 正常测试模型
        test_model()