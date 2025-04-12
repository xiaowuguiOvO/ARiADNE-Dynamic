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
                 map_size=10.0,                # 地图大小(正方形)
                 target_radius=0.5,            # 目标点半径
                 max_steps=1000,               # 最大步数
                 dt=0.1,                       # 时间步长
                 max_linear_velocity=1.0,      # 最大线速度
                 max_angular_velocity=1.0,     # 最大角速度
                 render_mode=None,             # 渲染模式
                 non_blocking_render=True):    # 是否使用非阻塞渲染（不抢占焦点）
        super().__init__()
        
        # 环境参数
        self.map_size = map_size
        self.target_radius = target_radius
        self.max_steps = max_steps
        self.dt = dt
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.render_mode = render_mode
        self.non_blocking_render = non_blocking_render
        
        # 状态空间：[x, y, theta, v_linear, v_angular, distance_to_target]
        self.observation_space = spaces.Dict({
            'robot_state': spaces.Box(
                low=np.array([-map_size/2, -map_size/2, -np.pi, 0, -max_angular_velocity, 0]),
                high=np.array([map_size/2, map_size/2, np.pi, max_linear_velocity, max_angular_velocity, np.sqrt(2)*map_size]),
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
        self.previous_distance_to_target = 0
        self.distance_to_target = 0
        
        # 渲染相关
        self.fig = None
        self.ax = None
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机初始化机器人位置和朝向
        self.total_reward = 0
        x = self.np_random.uniform(-self.map_size/2, self.map_size/2)
        y = self.np_random.uniform(-self.map_size/2, self.map_size/2)
        theta = self.np_random.uniform(-np.pi, np.pi)
        v_linear = 0.0
        v_angular = 0.0
        
        # 随机生成目标点(确保与机器人初始位置有一定距离)
        while True:
            target_x = self.np_random.uniform(-self.map_size/2, self.map_size/2)
            target_y = self.np_random.uniform(-self.map_size/2, self.map_size/2)
            target_pos = np.array([target_x, target_y], dtype=np.float32)
            
            # 计算与机器人的距离
            dist = np.linalg.norm(target_pos - np.array([x, y]))
            if dist > 3.0:  # 确保初始目标点与机器人有一定距离
                break
        
        self.target_position = target_pos
        
        # 计算到目标点的距离
        distance_to_target = np.linalg.norm(self.target_position - np.array([x, y]))
        self.distance_to_target = distance_to_target
        self.previous_distance_to_target = distance_to_target
        
        # 计算目标方向
        target_direction = np.arctan2(self.target_position[1] - y, self.target_position[0] - x)
        
        # 计算角度差 (目标方向 - 机器人朝向)
        heading_diff = np.arctan2(np.sin(target_direction - theta), np.cos(target_direction - theta))
        
        # 保存机器人的真实朝向（用于内部计算）
        self.robot_heading = theta
        
        # 更新机器人状态，使用角度差代替绝对朝向
        self.robot_state = np.array([x, y, heading_diff, v_linear, v_angular, distance_to_target], dtype=np.float32)
        
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
        done = False
        v_linear = np.clip(action[0], 0, self.max_linear_velocity)
        v_angular = np.clip(action[1], -self.max_angular_velocity, self.max_angular_velocity)
        action = np.array([v_linear, v_angular])
        
        # 更新机器人状态
        x, y, heading_diff, _, _, _ = self.robot_state
        
        # 更新机器人真实朝向（内部使用）
        self.robot_heading = self.robot_heading + v_angular * self.dt
        # 将角度标准化到[-pi, pi]
        self.robot_heading = np.arctan2(np.sin(self.robot_heading), np.cos(self.robot_heading))
        
        # 使用真实朝向更新位置
        x_new = x + v_linear * np.cos(self.robot_heading) * self.dt
        y_new = y + v_linear * np.sin(self.robot_heading) * self.dt
        
        # 边界处理
        x_new = np.clip(x_new, -self.map_size/2, self.map_size/2)
        y_new = np.clip(y_new, -self.map_size/2, self.map_size/2)
        
        # 计算与目标点的距离
        self.previous_distance_to_target = self.distance_to_target
        dist_to_target = np.linalg.norm(np.array([x_new, y_new]) - self.target_position)
        self.distance_to_target = dist_to_target
        
        # 计算目标方向
        target_direction = np.arctan2(self.target_position[1] - y_new, self.target_position[0] - x_new)
        
        # 计算新的角度差（目标方向 - 机器人朝向）
        heading_diff_new = np.arctan2(np.sin(target_direction - self.robot_heading), 
                                      np.cos(target_direction - self.robot_heading))
        
        # 更新状态，使用新的角度差
        self.robot_state = np.array([x_new, y_new, heading_diff_new, v_linear, v_angular, dist_to_target], dtype=np.float32)
        
        # 检查是否到达目标
        reached_target = self.distance_to_target <= self.target_radius
        
        # 计算奖励，基于接近程度和角度差减小程度
        approach_reward = self.previous_distance_to_target - self.distance_to_target
        angle_thresh = 0
        heading_reward = angle_thresh - abs(heading_diff_new)

        speed_reward = action[0] * 1.0
        # 计算总奖励
        reward = approach_reward + heading_reward + speed_reward
        
        if reached_target:
            reward += 100.0  # 到达目标的奖励
            done = True
            
        # 保存当前奖励和角度信息用于显示
        self.current_reward = f"{reward:.2f}"
        self.total_reward += reward
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
            'reached_target': reached_target,
            'heading_diff': heading_diff_new
        }
        
        if self.render_mode == 'human':
            self._render_frame()
            
        return observation, reward, terminated, truncated, info, done
    
    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            return self._render_frame()
    
    def _render_frame(self):
        """渲染当前帧"""
        if self.fig is None:
            # 设置matplotlib后端和关闭交互模式
            import matplotlib
            matplotlib.use('TkAgg')  # 使用TkAgg后端
            plt.ioff()  # 关闭交互模式，避免自动显示和抢占焦点
            
            # 创建图形，但不显示
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(-self.map_size/2 - 1, self.map_size/2 + 1)
            self.ax.set_ylim(-self.map_size/2 - 1, self.map_size/2 + 1)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            plt.title('Local Planner Environment')
            
            # 手动显示图形，但不激活窗口
            self.fig.show()
        
        self.ax.clear()
        self.ax.set_xlim(-self.map_size/2 - 1, self.map_size/2 + 1)
        self.ax.set_ylim(-self.map_size/2 - 1, self.map_size/2 + 1)
        self.ax.grid(True)
        
        # 绘制边界
        self.ax.plot([-self.map_size/2, self.map_size/2, self.map_size/2, -self.map_size/2, -self.map_size/2],
                     [-self.map_size/2, -self.map_size/2, self.map_size/2, self.map_size/2, -self.map_size/2],
                     'k-', linewidth=2)
        
        # 获取机器人信息
        x, y, heading_diff, v_linear, v_angular, distance_to_target = self.robot_state
        
        # 机器人圆形表示
        robot_circle = Circle((x, y), 0.3, color='blue', alpha=0.7)
        self.ax.add_patch(robot_circle)
        
        # 使用真实朝向绘制机器人方向
        length = 0.5
        self.ax.arrow(x, y, length * np.cos(self.robot_heading), length * np.sin(self.robot_heading),
                      head_width=0.2, head_length=0.2, fc='red', ec='red')
        
        # 绘制目标方向
        target_direction = np.arctan2(self.target_position[1] - y, self.target_position[0] - x)
        self.ax.arrow(x, y, length * np.cos(target_direction), length * np.sin(target_direction),
                      head_width=0.15, head_length=0.15, fc='green', ec='green')
        
        # 绘制目标点
        target_circle = Circle((self.target_position[0], self.target_position[1]), 
                               self.target_radius, color='green', alpha=0.7)
        self.ax.add_patch(target_circle)
        
        # 添加信息文本
        dist = np.linalg.norm(self.robot_state[:2] - self.target_position)
        
        # 如果还没有定义reward属性，初始化为N/A
        if not hasattr(self, 'current_reward'):
            self.current_reward = "N/A"
            
        heading_diff_deg = np.degrees(heading_diff)
        heading_deg = np.degrees(self.robot_heading)
        target_dir_deg = np.degrees(target_direction)
        
        info_text = f"step: {self.steps}\ntarget_dis: {dist:.2f}\nreward: {self.current_reward}\ntotal_reward: {self.total_reward:.2f}\nheading: {heading_deg:.2f}°\ntarget_dir: {target_dir_deg:.2f}°\nheading_diff: {heading_diff_deg:.2f}°\nlinear_vel: {v_linear:.2f}\nangular_vel: {v_angular:.2f}"
        self.ax.text(-self.map_size/2 + 0.5, self.map_size/2 - 1, info_text,
                     fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        # 使用canvas更新而不是plt.draw()和plt.pause()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        # 使用一个小延迟，但不会抢占焦点
        import time
        time.sleep(0.01)
        
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
        obs, reward, terminated, truncated, info, done = env.step(action)
        
        print(f"Reward: {reward}, Distance: {info['distance_to_target']:.2f}")
        
        if terminated or truncated:
            break
            
    env.close()


if __name__ == "__main__":
    test_env()