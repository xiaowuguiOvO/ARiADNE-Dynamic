import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import gymnasium as gym
from gymnasium import spaces
from parameter import *

class LocalPlannerEnv(gym.Env):
    """
    一个简单的本地规划器环境，用于测试dualstageagent的localcontroller。
    
    环境描述：
    - 在一个正方形的平面场景中随机生成目标点
    - 当agent到达目标点后，目标点会随机刷新到另一个位置
    """
    
    def __init__(self, 
                 map_size=20.0,                # 地图大小(正方形)
                 target_radius=0.5,            # 目标点半径
                 max_steps=1000,               # 最大步数
                 dt=0.1,                       # 时间步长
                 max_linear_velocity=2.0,      # 最大线速度
                 max_angular_velocity=1.0,     # 最大角速度
                 render_mode=None):            # 渲染模式
        super().__init__()
        
        # 环境参数
        self.map_size = map_size
        self.target_radius = target_radius
        self.max_steps = max_steps
        self.dt = dt
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.render_mode = render_mode
        
        # 状态空间：[x, y, theta, v_linear, v_angular]
        self.observation_space = spaces.Dict({
            'robot_state': spaces.Box(
                low=np.array([-map_size/2, -map_size/2, -np.pi, 0, -max_angular_velocity]),
                high=np.array([map_size/2, map_size/2, np.pi, max_linear_velocity, max_angular_velocity]),
                dtype=np.float32
            ),
            'target_position': spaces.Box(
                low=np.array([-map_size/2, -map_size/2]),
                high=np.array([map_size/2, map_size/2]),
                dtype=np.float32
            )
        })
        
        # 动作空间：[线速度, 角速度]
        self.action_space = spaces.Box(
            low=np.array([0, -max_angular_velocity]),
            high=np.array([max_linear_velocity, max_angular_velocity]),
            dtype=np.float32
        )
        
        # 初始化状态
        self.robot_state = None
        self.target_position = None
        self.steps = 0
        
        # 渲染相关
        self.fig = None
        self.ax = None
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机初始化机器人位置和朝向
        x = self.np_random.uniform(-self.map_size/2, self.map_size/2)
        y = self.np_random.uniform(-self.map_size/2, self.map_size/2)
        theta = self.np_random.uniform(-np.pi, np.pi)
        v_linear = 0.0
        v_angular = 0.0
        
        self.robot_state = np.array([x, y, theta, v_linear, v_angular], dtype=np.float32)
        
        # 随机生成目标点(确保与机器人初始位置有一定距离)
        while True:
            target_x = self.np_random.uniform(-self.map_size/2, self.map_size/2)
            target_y = self.np_random.uniform(-self.map_size/2, self.map_size/2)
            target_pos = np.array([target_x, target_y], dtype=np.float32)
            
            # 计算与机器人的距离
            dist = np.linalg.norm(target_pos - self.robot_state[:2])
            if dist > 3.0:  # 确保初始目标点与机器人有一定距离
                break
        
        self.target_position = target_pos
        self.steps = 0
        
        observation = {
            'robot_state': self.robot_state,
            'target_position': self.target_position
        }
        
        info = {}
        
        if self.render_mode == 'human':
            self._render_frame()
            
        return observation, info
    
    def step(self, action):
        """执行动作并更新环境"""
        # 确保动作在合法范围内
        v_linear = np.clip(action[0], 0, self.max_linear_velocity)
        v_angular = np.clip(action[1], -self.max_angular_velocity, self.max_angular_velocity)
        action = np.array([v_linear, v_angular])
        
        # 更新机器人状态
        x, y, theta, _, _ = self.robot_state
        
        # 运动学模型更新位置和朝向
        theta_new = theta + v_angular * self.dt
        # 将角度标准化到[-pi, pi]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        x_new = x + v_linear * np.cos(theta_new) * self.dt
        y_new = y + v_linear * np.sin(theta_new) * self.dt
        
        # 边界处理
        x_new = np.clip(x_new, -self.map_size/2, self.map_size/2)
        y_new = np.clip(y_new, -self.map_size/2, self.map_size/2)
        
        # 更新状态
        self.robot_state = np.array([x_new, y_new, theta_new, v_linear, v_angular], dtype=np.float32)
        
        # 计算与目标点的距离
        dist_to_target = np.linalg.norm(self.robot_state[:2] - self.target_position)
        
        # 检查是否到达目标
        reached_target = dist_to_target <= self.target_radius
        
        # 计算奖励
        reward = -0.1  # 每步的小惩罚，鼓励快速到达目标
        
        if reached_target:
            reward += 10.0  # 到达目标的奖励
            # 更新目标点位置
            while True:
                new_target_x = self.np_random.uniform(-self.map_size/2, self.map_size/2)
                new_target_y = self.np_random.uniform(-self.map_size/2, self.map_size/2)
                new_target_pos = np.array([new_target_x, new_target_y], dtype=np.float32)
                
                # 确保新目标与当前位置有一定距离
                dist = np.linalg.norm(new_target_pos - self.robot_state[:2])
                if dist > 3.0:
                    break
                    
            self.target_position = new_target_pos
        
        # 更新步数并检查是否结束
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps
        
        # 准备观测和信息
        observation = {
            'robot_state': self.robot_state,
            'target_position': self.target_position
        }
        
        info = {
            'distance_to_target': dist_to_target,
            'reached_target': reached_target
        }
        
        if self.render_mode == 'human':
            self._render_frame()
            
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            return self._render_frame()
    
    def _render_frame(self):
        """渲染当前帧"""
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(-self.map_size/2 - 1, self.map_size/2 + 1)
            self.ax.set_ylim(-self.map_size/2 - 1, self.map_size/2 + 1)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            plt.title('Local Planner Environment')
        
        self.ax.clear()
        self.ax.set_xlim(-self.map_size/2 - 1, self.map_size/2 + 1)
        self.ax.set_ylim(-self.map_size/2 - 1, self.map_size/2 + 1)
        self.ax.grid(True)
        
        # 绘制边界
        self.ax.plot([-self.map_size/2, self.map_size/2, self.map_size/2, -self.map_size/2, -self.map_size/2],
                     [-self.map_size/2, -self.map_size/2, self.map_size/2, self.map_size/2, -self.map_size/2],
                     'k-', linewidth=2)
        
        # 绘制机器人
        x, y, theta, v_linear, v_angular = self.robot_state
        
        # 机器人圆形表示
        robot_circle = Circle((x, y), 0.3, color='blue', alpha=0.7)
        self.ax.add_patch(robot_circle)
        
        # 机器人朝向
        length = 0.5
        self.ax.arrow(x, y, length * np.cos(theta), length * np.sin(theta),
                      head_width=0.2, head_length=0.2, fc='red', ec='red')
        
        # 绘制目标点
        target_circle = Circle((self.target_position[0], self.target_position[1]), 
                               self.target_radius, color='green', alpha=0.7)
        self.ax.add_patch(target_circle)
        
        # 添加信息文本
        dist = np.linalg.norm(self.robot_state[:2] - self.target_position)
        info_text = f"步数: {self.steps}\n到目标距离: {dist:.2f}"
        self.ax.text(-self.map_size/2 + 0.5, self.map_size/2 - 1, info_text,
                     fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.draw()
        plt.pause(0.001)
        
    def close(self):
        """关闭环境"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def test_env():
    """测试环境功能"""
    env = LocalPlannerEnv(render_mode='human')
    obs, info = env.reset()
    
    for _ in range(100):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward}, Distance: {info['distance_to_target']:.2f}")
        
        if terminated or truncated:
            break
            
    env.close()


if __name__ == "__main__":
    test_env() 