import numpy as np
import time
from local_planner_env import LocalPlannerEnv
from dual_stage_agent import DualStageAgent
import torch
def test_env_only(max_steps=200, render=True):
    """
    仅测试LocalPlannerEnv环境的基本功能
    
    参数:
        max_steps: 测试的最大步数
        render: 是否渲染环境
    """
    # 创建环境
    render_mode = 'human' if render else None
    env = LocalPlannerEnv(
        map_size=20.0,
        target_radius=0.5,
        max_steps=max_steps,
        render_mode=render_mode
    )
    
    # 重置环境
    obs, info = env.reset()
    print(f"初始观察值: {obs}")
    print(f"初始信息: {info}")
    
    # 简单循环，使用固定的前进动作
    for step in range(max_steps):
        # 简单的动作：小速度前进并缓慢转向
        action = np.array([0.5, 0.1])  # [线速度, 角速度]
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 打印信息
        print(f"步骤 {step+1}:")
        print(f"  动作: [{action[0]:.2f}, {action[1]:.2f}]")
        print(f"  距离目标: {info['distance_to_target']:.2f}")
        print(f"  奖励: {reward:.2f}")
        
        # 检查是否到达目标
        if info['reached_target']:
            print(f"成功到达目标! 新目标位置: [{obs['target_position'][0]:.2f}, {obs['target_position'][1]:.2f}]")
        
        # 检查是否结束
        if terminated or truncated:
            print("回合结束")
            break
        
        # 控制渲染速度
        if render:
            time.sleep(0.1)
    
    # 关闭环境
    env.close()
    print("测试完成")

def test_local_controller(max_steps=200, render=True):
    """
    测试LocalController的控制功能
    """
    # 创建环境
    render_mode = 'human' if render else None
    env = LocalPlannerEnv(
        map_size=20.0,
        target_radius=0.5,
        max_steps=max_steps,
        render_mode=render_mode
    )
    agent = DualStageAgent(device='cpu')
    obs, info = env.reset()
    print(f"init obs: {obs}")
    x, y, theta, v_linear, v_angular, distance_to_target = obs['robot_state']
    input_robot_state = np.array([distance_to_target, theta, v_linear, v_angular], dtype=np.float32)
    input_robot_state = torch.tensor(input_robot_state, dtype=torch.float32)
    print(f"input_robot_state: {input_robot_state}")
    for step in range(max_steps):
        action = agent.local_controller(input_robot_state)
        action = action.detach().numpy()
        action = np.array([1, 0.1])
        print(f"action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

if __name__ == "__main__":
    # 运行测试
    # test_env_only(max_steps=300, render=True)
    test_local_controller(max_steps=300, render=True)