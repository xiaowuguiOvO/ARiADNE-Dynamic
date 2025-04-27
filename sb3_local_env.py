import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy
from io import BytesIO
from PIL import Image
from sensor import sensor_work
from utils import *
import random
from parameter import *
import torch
import math
import gymnasium as gym   # 如果你在训练脚本里用的是 gymnasium

class Env:
    def __init__(self, episode_index, plot=False, random_wapoint=False, render_mode=None):
        self.episode_index = episode_index
        self.plot = plot
        self.ground_truth, self.robot_cell = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.cell_size = CELL_SIZE  # meter
        self.random_wapoint = random_wapoint # 是否随机生成目标点
        self.agent = None
        self.robot_location = np.array([0.0, 0.0])  # meter
        self.robot_belief = np.ones(self.ground_truth_size) * 127
        self.belief_origin_x = -np.round(self.robot_cell[0] * self.cell_size, 1)   # meter
        self.belief_origin_y = -np.round(self.robot_cell[1] * self.cell_size, 1)  # meter
        self.render_mode = render_mode
        self.global_frontiers = set()

        self.sensor_range = SENSOR_RANGE  # meter
        self.travel_dist = 0  # meter
        self.explored_rate = 0

        self.robot_belief = sensor_work(self.robot_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                        self.ground_truth)
        self.old_belief = deepcopy(self.robot_belief)

        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        if self.plot:
            self.frame_files = []
            self.trajectory_x = [self.robot_location[0]]
            self.trajectory_y = [self.robot_location[1]]
            
        # 动态障碍物相关
        self.dynamic_obstacles = []
        self.collision_count = 0
        self.init_dynamic_obstacles()

        self.step_size = STEP_SIZE
        self.decision_interval = DECISION_INTERVAL
        self.decision_distance = DECISION_DISTANCE
        self.waypoint_threshold = WAYPOINT_THRESHOLD

        
        self.velocity_command = np.array([0.0, 0.0])
        self.total_reward = 0.0

        self.previous_distance_to_target = 0.0
        self.distance_to_target = 0.0
        self.heading_diff = 0.0
        
        # 关于step
        self.step_count = 0
        self.max_steps = MAX_EPISODE_STEP

        self.fig = None
        self.ax = None
        self.ax_truth = None
        self.ax_updating = None
        self.im = None
        self.im_truth = None
        self.im_updating = None
        self.robot_point_updating = None
        self.waypoint_point_updating = None
        self.heading_arrow_updating = None
        self.frontier_points_updating = None
        self.ray_lines = []
        self.previous_v_angular = 0.0
        self.previous_v_linear = 0.0
        
    def reset(self):
        # self.episode_index = np.random.randint(1, 5000)
        self.step_count = 0
        self.total_reward = 0.0
        self.agent.waypoint = None
        # 随机 1 - 5001
        self.episode_index = np.random.randint(1, 5001)
        # self.episode_index = 5
        self.ground_truth, self.robot_cell = self.import_ground_truth(self.episode_index)
        self.belief_origin_x = -np.round(self.robot_cell[0] * self.cell_size, 1)   # meter
        self.belief_origin_y = -np.round(self.robot_cell[1] * self.cell_size, 1) 
        
        self.robot_location = np.array([
        self.robot_cell[0] * self.cell_size + self.belief_origin_x,
        self.robot_cell[1] * self.cell_size + self.belief_origin_y
        ])
        self.agent.location = self.robot_location
        
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.robot_belief = np.ones(self.ground_truth_size) * 127
        self.robot_belief = sensor_work(self.robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth)
        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        obs = self.agent.get_robot_state()
        
        # reset render
        if self.render_mode == 'human':
            self._reset_render()
        
        # random waypoint
        if self.random_wapoint:
            _, self.agent.waypoint = self.generate_random_waypoint()
        return obs, {}
    
    def set_agent(self, agent):
        self.agent = agent

    def init_dynamic_obstacles(self):
        """在自由空间中随机放置动态障碍物,每个障碍物在两点之间往返运动"""
        # 找出所有自由空间单元格
        free_cells = np.argwhere(self.ground_truth == FREE)
        
        # 移除机器人附近的单元格
        robot_pos = np.array([self.robot_cell[1], self.robot_cell[0]])
        distances = np.linalg.norm(free_cells - robot_pos, axis=1)
        mask = distances > (SENSOR_RANGE / self.cell_size) * 0.3
        free_cells = free_cells[mask]
        
        # 如果没有足够的自由空间，减少障碍物数量
        obstacle_count = min(NUM_DYNAMIC_OBSTACLES, len(free_cells) // 20)
        
        if len(free_cells) > 0:
            for i in range(obstacle_count):
                # 为每个障碍物随机选择两个端点
                valid_path = False
                attempts = 0
                while not valid_path and attempts < 50:  # 添加最大尝试次数
                    attempts += 1
                    # 随机选择第一个点
                    idx1 = np.random.randint(len(free_cells))
                    y1, x1 = free_cells[idx1]
                    pos1 = np.array([
                        x1 * self.cell_size + self.belief_origin_x,
                        y1 * self.cell_size + self.belief_origin_y
                    ])
                    
                    # 随机选择轨迹距离
                    min_dist = np.random.uniform(OBSTACLE_CURVE_MIN_DIST, OBSTACLE_CURVE_MAX_DIST) / self.cell_size
                    max_dist = min_dist + 5.0 / self.cell_size  # 增加一个小范围以确保合理性
                    
                    nearby_cells = []
                    for j, (y2, x2) in enumerate(free_cells):
                        dist = np.linalg.norm([y2-y1, x2-x1])
                        if min_dist < dist < max_dist:
                            # 检查路径上是否有障碍物
                            if self.check_path_valid(x1, y1, x2, y2):
                                nearby_cells.append((y2, x2))
                    
                    if nearby_cells:
                        # 随机选择一个有效的第二个点
                        y2, x2 = nearby_cells[np.random.randint(len(nearby_cells))]
                        pos2 = np.array([
                            x2 * self.cell_size + self.belief_origin_x,
                            y2 * self.cell_size + self.belief_origin_y
                        ])
                        valid_path = True
                
                if valid_path:  # 只有找到有效路径才创建障碍物
                    # 计算初始位置和速度
                    direction = pos2 - pos1
                    speed = np.random.uniform(MIN_OBSTACLE_SPEED, MAX_OBSTACLE_SPEED)  # 随机速度
                    velocity = direction / np.linalg.norm(direction) * speed
                    
                    obstacle = {
                        'id': i,
                        'position': pos1.copy(),
                        'velocity': velocity,
                        'radius': OBSTACLE_RADIUS,
                        'waypoint1': pos1,
                        'waypoint2': pos2,
                        'current_target': pos2,
                        'speed': speed
                    }
                    
                    self.dynamic_obstacles.append(obstacle)

    def check_path_valid(self, x1, y1, x2, y2):
        """检查两点之间的路径是否有障碍物"""
        # Bresenham算法检查路径
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if not (0 <= y < self.ground_truth_size[0] and 
                    0 <= x < self.ground_truth_size[1]):
                return False
            if self.ground_truth[int(y), int(x)] != FREE:
                return False
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return True

    def update_dynamic_obstacles(self, dt=0.1):
        """更新所有动态障碍物的位置"""
        for i, obs in enumerate(self.dynamic_obstacles):
            # 保存旧位置用于调试
            old_pos = obs['position'].copy()
            
            # 检查是否到达当前目标点
            dist_to_target = np.linalg.norm(obs['current_target'] - obs['position'])
            
            if dist_to_target < obs['speed'] * dt:
                # 到达目标点，切换目标
                if np.array_equal(obs['current_target'], obs['waypoint1']):
                    obs['current_target'] = obs['waypoint2']
                else:
                    obs['current_target'] = obs['waypoint1']
                
                # 更新速度方向
                direction = obs['current_target'] - obs['position']
                obs['velocity'] = direction / np.linalg.norm(direction) * obs['speed']
            
            # 更新位置
            obs['position'] += obs['velocity'] * dt
            
            # 打印位置变化，用于调试
            # print(f"障碍物{i} 从 {old_pos} 移动到 {obs['position']}, 移动了 {np.linalg.norm(obs['position']-old_pos):.4f}m")

    
    def import_ground_truth(self, episode_index):
        map_dir = f'maps'
        map_list = os.listdir(map_dir)
        map_index = episode_index % np.size(map_list)
        ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1) * 255).astype(int)

        ground_truth = block_reduce(ground_truth, 2, np.min)

        robot_cell = np.nonzero(ground_truth == 208)
        robot_cell = np.array([np.array(robot_cell)[1, 10], np.array(robot_cell)[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell

    def update_robot_location(self, robot_location):
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])
        if self.plot:
            self.trajectory_x.append(self.robot_location[0])
            self.trajectory_y.append(self.robot_location[1])

    def update_robot_belief(self):
        self.robot_belief = sensor_work(self.robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                            self.ground_truth)
        
    def _get_static_obstacle_reward(self, robot_belief, num_rays=30, fov_deg=240, step_size=1):
        """
        加速版：使用射线投射方式计算静态障碍物奖励

        参数:
            robot_belief: torch.Tensor, shape=(H, W)，值为 {0, 127, 255}
            num_rays: int，射线数量
            fov_deg: float，激光雷达视角范围 (单位：度)
            step_size: float，每次步进的像素距离

        返回:
            reward: torch.Tensor
            rays: List[List[(x, y)]] 每条射线的路径，用于可视化
        """
        device = robot_belief.device
        H, W = robot_belief.shape
        cx, cy = W // 2, H // 2
        max_radius = min(H, W) // 2
        
        robot_heading = self.agent.heading_theta if hasattr(self.agent, 'heading_theta') else 0.0
        fov_rad = math.radians(fov_deg)
        # start_angle = -fov_rad / 2
        start_angle = robot_heading - fov_rad / 2
        angle_step = fov_rad / num_rays

        # 预计算角度单位向量
        directions = [(math.cos(start_angle + i * angle_step), math.sin(start_angle + i * angle_step))
                    for i in range(num_rays)]

        rays = []
        distances = []

        for dx, dy in directions:
            ray = []
            hit = False

            for s in range(1, max_radius):
                x = int(cx + dx * s * step_size)
                y = int(cy + dy * s * step_size)

                if x < 0 or x >= W or y < 0 or y >= H:
                    break

                ray.append((x, y))

                if robot_belief[y, x] == ROBOT_BELIEF_OCCUPIED:
                    distances.append(s * step_size)
                    hit = True
                    break

            if not hit:
                distances.append(max_radius)

            rays.append(ray)

        avg_dist = sum(distances) / len(distances)
        reward = torch.log(torch.tensor(avg_dist, dtype=torch.float32, device=device).clamp(min=1e-6))

        return reward, rays


    
    def calculate_reward(self):
        "local controller reward"
        reward = 0
        
        r_approach = 5
        r_heading = 1
        r_speed = 1
        r_static = 0.01
        r_smooth_linear = -0.1
        r_smooth_angular = -1
        # 这个smoth 的参数好像不对 加上去就寄了
        
        self.previous_distance_to_target = self.distance_to_target
        self.distance_to_target = self.agent.distance_to_target
        heading_diff = self.agent.heading_theta_diff
        
        current_v_linear = self.agent.v_linear
        current_v_angular = self.agent.v_angular
        # 计算速度变化 (需要确保 self.previous_v_linear/angular 在 step 中被正确更新)
        delta_v_linear = abs(current_v_linear - getattr(self, 'previous_v_linear', current_v_linear)) # 使用 getattr 提供默认值以防首次调用
        delta_v_angular = abs(current_v_angular - getattr(self, 'previous_v_angular', current_v_angular))
        
        approach_reward = r_approach * (self.previous_distance_to_target - self.distance_to_target)
        heading_reward = r_heading * ((np.pi / 12) - abs(heading_diff))
        # heading_reward = r_heading * np.cos(heading_diff) # 改用余弦奖励
        # heading_reward = r_heading * np.cos(heading_diff) # 直接使用角度差的余弦值
        speed_reward = r_speed * self.agent.v_linear
        static_reward, self.ray_lines = self._get_static_obstacle_reward(torch.from_numpy(self.agent.updating_map_info.map))
        static_reward = static_reward * r_static
        linear_smooth_penalty = r_smooth_linear * delta_v_linear
        angular_smooth_penalty = r_smooth_angular * delta_v_angular
        # print(f"{approach_reward:.2f}, {heading_reward:.2f}, {speed_reward:.2f}, {static_reward:.2f}, {linear_smooth_penalty:.2f}, {angular_smooth_penalty:.2f}")
        reward = approach_reward + heading_reward + speed_reward + static_reward
        
        self.previous_v_linear = current_v_linear
        self.previous_v_angular = current_v_angular
        return reward

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)
        
    def check_collision(self):
        """检查机器人是否与任何动态障碍物发生碰撞"""
        for obs in self.dynamic_obstacles:
            dist = np.linalg.norm(self.robot_location - obs['position'])
            if dist < (OBSTACLE_RADIUS + ROBOT_RADIUS):
                self.collision_count += 1
                return True
        return False
    
    def check_wall_collision(self, position):
        """检查给定位置是否与墙壁碰撞，使用ground_truth"""
        # 确保位置是整数坐标
        x, y = np.round((position - np.array([self.belief_origin_x, self.belief_origin_y])) / self.cell_size).astype(int)
        # 检查是否超出地图边界
        if x < 0 or x >= self.ground_truth.shape[1] or y < 0 or y >= self.ground_truth.shape[0]:
            return True
        return self.ground_truth[y, x] != GROUND_TRUTH_FREE
        
    def generate_random_waypoint(self):
        """
        在机器人指定范围内的自由空间随机生成一个目标点
        
        Returns:
            bool: 是否成功生成目标点
            np.array: 生成的目标点坐标
        """
        # 转换距离从米到像素
        max_dist_px = int(RANDOM_MAX_DIST / self.cell_size)
        min_dist_px = int(RANDOM_MIN_DIST / self.cell_size)
        
        # 获取当前机器人位置（像素坐标）
        robot_x_px = int((self.robot_location[0] - self.belief_origin_x) / self.cell_size)
        robot_y_px = int((self.robot_location[1] - self.belief_origin_y) / self.cell_size)
        
        # 获取地图尺寸
        H, W = self.ground_truth.shape
        
        # 最大尝试次数
        max_attempts = 100
        
        for _ in range(max_attempts):
            # 在圆环内随机生成点
            angle = np.random.uniform(0, 2*np.pi)
            # 使用sqrt确保点在圆环内均匀分布
            distance = np.random.uniform(min_dist_px**2, max_dist_px**2)**0.5
            
            # 计算相对偏移
            dx = int(distance * np.cos(angle))
            dy = int(distance * np.sin(angle))
            
            # 计算随机点坐标（像素坐标）
            x_px = robot_x_px + dx
            y_px = robot_y_px + dy
            
            # 检查点是否在地图内
            if 0 <= x_px < W and 0 <= y_px < H:
                # 检查点是否在自由空间
                if self.ground_truth[y_px, x_px] == FREE:
                    # 将像素坐标转换回米
                    x_m = x_px * self.cell_size + self.belief_origin_x
                    y_m = y_px * self.cell_size + self.belief_origin_y
                    return True, np.array([x_m, y_m])
        
        # 如果无法找到自由空间的点，返回机器人当前位置
        print(f"在距离{RANDOM_MIN_DIST}米到{RANDOM_MAX_DIST}米范围内没有找到合适的点")
        return False, self.robot_location
    
    def step(self, action):
        """
        执行一个模拟步骤，使用agent的当前速度（线速度和角速度）
        返回:
            reward: 奖励值
            collision: 是否发生碰撞
            need_decision: 是否需要做新决策
        """
        done = False
        terminated = False
        truncated = False
        wall_collision = False
        if self.agent is None:
            raise ValueError("必须先使用set_agent设置代理")
        # print(action)
        linear_vel = action[0]
        angular_vel = action[1]
        velocity_command = np.array([linear_vel, angular_vel])
        # print(f"velocity_command: {velocity_command}")
        # 初始化机器人朝向(如果不存在)
        if not hasattr(self, 'robot_orientation'):
            self.robot_orientation = 0.0  # 初始朝向
        
        # 更新朝向
        self.robot_orientation += angular_vel * self.step_size
        # 标准化到 [-π, π]
        self.robot_orientation = np.arctan2(np.sin(self.robot_orientation), np.cos(self.robot_orientation))
        
        # 更新agent velocity 和 heading_theta
        self.agent.v_linear = linear_vel
        self.agent.v_angular = angular_vel
        self.agent.heading_theta = self.robot_orientation
        # 使用线速度和朝向计算x,y方向的速度分量
        vx = linear_vel * np.cos(self.robot_orientation)
        vy = linear_vel * np.sin(self.robot_orientation)
        cartesian_velocity = np.array([vx, vy])
        
        # 使用笛卡尔速度计算下一个位置
        current_pos = self.robot_location.copy()
        next_pos = current_pos + cartesian_velocity * self.step_size
        
        dynamic_collision = False
        # print(f"wall_collision: {wall_collision}")
        
        # 保存原始控制命令用于记录
        self.velocity_command = velocity_command  # [linear, angular]
        # 保存转换后的笛卡尔速度用于其他计算
        self.cartesian_velocity = cartesian_velocity
        
        # 记录总移动距离和是否碰撞
        total_dist = 0.0
        collision = False
        need_decision = False

        old_location = self.robot_location.copy()
        # 更新位置
        self.robot_location = next_pos
        self.agent.location = self.robot_location
        # if not wall_collision:
        #     self.robot_location = next_pos
        # else:
        #     self.robot_location = old_location - cartesian_velocity * self.step_size * 0.5
            # print("collision, old belief: ", self.robot_belief[self.robot_cell[1], self.robot_cell[0]])
        # 更新栅格位置
        self.robot_cell = np.round(
            np.array([(self.robot_location[0] - self.belief_origin_x) / self.cell_size,
                        (self.robot_location[1] - self.belief_origin_y) / self.cell_size])
        ).astype(int)
        
        # 计算agent的状态  dis_to_target
        self.agent.distance_to_target = self.agent.cal_dist_to_waypoint(self.agent.waypoint)
        self.agent.heading_theta_diff = self.agent.cal_heading_theta_diff_to_waypoint(self.agent.waypoint)
        
        # 更新移动距离
        moved_dist = np.linalg.norm(self.robot_location - old_location)
        total_dist += moved_dist
        # 更新动态障碍物位置
        # self.update_dynamic_obstacles(self.step_size)
        
        if self.check_wall_collision(self.robot_location):
            wall_collision = True
        
        
        
        # 更新robot belief
        self.update_robot_belief()
        self.agent.belief_info = self.belief_info
        self.agent.update_map(self.belief_info)
        self.agent.update_updating_map(self.agent.location)
        self.agent.update_frontiers()
        # self.agent.update_local_belief_map()
        # 累计总移动距离
        self.travel_dist += total_dist
        # 评估探索率
        self.evaluate_exploration_rate()
        # 计算奖励
        reward = self.calculate_reward()
        self.total_reward += reward
        
        if wall_collision:
            terminated = True
            reward -= WALL_COLLISION_PENALTY
        # check is arrive
        if self.agent.check_arrive_waypoint(self.agent.waypoint):
            terminated = True
            reward += REACH_WAYPOINT_REWARD
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
            terminated = True
            reward -= 100.0
        done = terminated or truncated
        
        # print(self.step_count)
        obs = self.agent.get_robot_state()
        if self.plot:
            self.plot_env(self.step_count)
            self.agent.plot_env(self.agent.waypoint)
            if done:
                make_gif(gifs_path, self.step_count, self.frame_files, self.explored_rate)
                
        return obs, reward, terminated, truncated, {}
    
    def _reset_render(self):
        # 清除之前的图像
        if hasattr(self, 'fig') and self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.ax_truth = None
            self.ax_updating = None
            self.im = None
            self.im_truth = None
            self.im_updating = None
            self.robot_point_updating = None
            self.waypoint_point_updating = None
            self.heading_arrow_updating = None
            self.frontier_points_updating = None

    def _render_belief_map(self):
        """渲染左侧的belief map及其相关元素"""
        if not hasattr(self, 'im') or self.im is None:
            # 初始化左边子图 - robot_belief
            self.im = self.ax.imshow(self.robot_belief, cmap='gray', origin='lower')
            self.robot_point, = self.ax.plot([], [], 'mo', markersize=5, zorder=5)
            self.waypoint_point = self.ax.scatter([], [], c='blue', s=5, marker='*', zorder=5)
            self.heading_arrow = self.ax.quiver([], [], [], [], color='red', scale=20, zorder=6)
            self.frontier_points = self.ax.scatter([], [], c='red', s=4, marker='.', zorder=4)
            self.info_text = self.ax.text(0.02, 1.05, '', transform=self.ax.transAxes)
            self.updating_map_rect = plt.Rectangle((0, 0), 1, 1, fill=False, color='green', linewidth=2, zorder=7)
            self.ax.add_patch(self.updating_map_rect)
        else:
            # 更新belief map
            self.im.set_data(self.robot_belief)
        
        # 更新机器人位置
        robot_x = (self.robot_location[0] - self.belief_origin_x) / self.cell_size
        robot_y = (self.robot_location[1] - self.belief_origin_y) / self.cell_size
        self.robot_point.set_data([robot_x], [robot_y])
        
        # 更新朝向箭头
        if hasattr(self.agent, 'heading_theta'):
            dx = np.cos(self.agent.heading_theta)
            dy = np.sin(self.agent.heading_theta)
            self.heading_arrow.set_offsets([[robot_x, robot_y]])
            self.heading_arrow.set_UVC(dx, dy)
        
        # 更新waypoint位置
        if hasattr(self.agent, 'waypoint') and self.agent.waypoint is not None:
            waypoint_x = (self.agent.waypoint[0] - self.belief_origin_x) / self.cell_size
            waypoint_y = (self.agent.waypoint[1] - self.belief_origin_y) / self.cell_size
            self.waypoint_point.set_offsets([[waypoint_x, waypoint_y]])
        else:
            self.waypoint_point.set_offsets(np.array([[]], dtype=float).reshape(0, 2))
        
        # 更新前沿点
        if hasattr(self.agent, 'frontier') and len(self.agent.frontier) > 0:
            frontier_coords = np.array(list(self.agent.frontier))
            frontier_x = (frontier_coords[:, 0] - self.belief_origin_x) / self.cell_size
            frontier_y = (frontier_coords[:, 1] - self.belief_origin_y) / self.cell_size
            frontier_points = np.column_stack((frontier_x, frontier_y))
            self.frontier_points.set_offsets(frontier_points)
            self.frontier_points.set_visible(True)
            self.frontier_points.set_zorder(5)
            self.frontier_points.set_sizes([1])
            self.frontier_points.set_color('red')
        else:
            self.frontier_points.set_offsets(np.array([[]], dtype=float).reshape(0, 2))
            
        if hasattr(self.agent, 'updating_map_size'):
            # 获取地图尺寸
            map_height, map_width = self.robot_belief.shape
            # 计算updating_map的大小（栅格单位）
            size_in_cells = int(self.agent.updating_map_size / self.cell_size)
            # print(self.agent.updating_map_size, self.cell_size, size_in_cells)
            # 计算矩形框的位置，确保完全在地图范围内
            half_size = size_in_cells // 2
            rect_x = np.clip(robot_x - half_size, 0, map_width - size_in_cells)
            rect_y = np.clip(robot_y - half_size, 0, map_height - size_in_cells)
            # 打印调试信息
            # print(f"Map size: {map_width}x{map_height}, Robot pos: ({robot_x:.2f}, {robot_y:.2f})")
            # print(f"Rect pos: ({rect_x:.2f}, {rect_y:.2f}), size: {size_in_cells}")
            # 更新矩形框
            self.updating_map_rect.set_xy((rect_x, rect_y))
            self.updating_map_rect.set_width(size_in_cells)
            self.updating_map_rect.set_height(size_in_cells)
            self.updating_map_rect.set_visible(True)
            # 设置矩形框的样式
            self.updating_map_rect.set_facecolor('none')  # 透明填充
            self.updating_map_rect.set_edgecolor('green')  # 绿色边框
            self.updating_map_rect.set_linewidth(1)  # 设置线宽
            # 强制重绘
            self.fig.canvas.draw_idle()
            
            # 更新信息文本
            info_str = f'v_lin: {self.agent.v_linear:.2f}  v_ang: {self.agent.v_angular:.2f}  reward: {self.total_reward:.2f} step: {self.step_count} dis: {self.agent.distance_to_target:.2f} x: {self.robot_location[0]:.2f} y: {self.robot_location[1]:.2f} c_x: {self.robot_cell[0]} c_y: {self.robot_cell[1]}'
            self.info_text.set_text(info_str)
            self.ax.axis('off')

    def _render_ground_truth(self):
        """渲染右侧的ground truth地图及其相关元素"""
        if not hasattr(self, 'im_truth') or self.im_truth is None:
            # 初始化右边子图 - ground_truth
            self.im_truth = self.ax_truth.imshow(self.ground_truth, cmap='gray', origin='lower')
            self.robot_point_truth, = self.ax_truth.plot([], [], 'mo', markersize=5, zorder=5)
            self.waypoint_point_truth = self.ax_truth.scatter([], [], c='blue', s=5, marker='*', zorder=5)  # 修正：在ax_truth上创建waypoint
        else:
            # 更新ground truth
            self.im_truth.set_data(self.ground_truth)
        
        # 更新机器人位置
        robot_x = (self.robot_location[0] - self.belief_origin_x) / self.cell_size
        robot_y = (self.robot_location[1] - self.belief_origin_y) / self.cell_size
        self.robot_point_truth.set_data([robot_x], [robot_y])

        # 更新waypoint位置 - 使用相同的坐标转换方式
        if hasattr(self.agent, 'waypoint') and self.agent.waypoint is not None:
            waypoint_x = (self.agent.waypoint[0] - self.belief_origin_x) / self.cell_size
            waypoint_y = (self.agent.waypoint[1] - self.belief_origin_y) / self.cell_size
            self.waypoint_point_truth.set_offsets([[waypoint_x, waypoint_y]])
        else:
            self.waypoint_point_truth.set_offsets(np.array([[]], dtype=float).reshape(0, 2))
            
        self.ax_truth.axis('off')

    def _render_updating_belief_map(self):
        """渲染更新中的局部belief map"""
        # 检查agent和updating_map_info
        if not hasattr(self, 'agent') or self.agent is None:
            return
        
        try:
            if not hasattr(self.agent, 'updating_map_info') or self.agent.updating_map_info is None:
                return  # 如果agent没有updating_map_info属性，直接返回
                
            if not hasattr(self, 'im_updating') or self.im_updating is None:
                # 初始化更新belief map的子图
                # print(self.agent.updating_map_info.map)
                self.im_updating = self.ax_updating.imshow(
                    self.agent.updating_map_info.map, 
                    cmap='gray', 
                    origin='lower',
                    vmin=0,    # 设置颜色映射的最小值
                    vmax=255   # 设置颜色映射的最大值
                )
                self.robot_point_updating, = self.ax_updating.plot([], [], 'mo', markersize=5, zorder=5)
                self.waypoint_point_updating = self.ax_updating.scatter([], [], c='blue', s=5, marker='*', zorder=5)
                self.heading_arrow_updating = self.ax_updating.quiver([], [], [], [], color='red', scale=20, zorder=6)
                self.frontier_points_updating = self.ax_updating.scatter([], [], c='red', s=4, marker='.', zorder=4)
                self.info_text_updating = self.ax_updating.text(0.02, 1.05, 'Updating Belief Map', transform=self.ax_updating.transAxes)
                # self.ax_updating.set_title('Robot Updating Belief Map')
                self.ray_lines_updating = []
            else:
                # 更新belief map
                self.im_updating.set_data(self.agent.updating_map_info.map)
                self.im_updating.set_clim(0, 255)
            # 获取updating map的坐标原点和尺寸
            updating_origin_x = self.agent.updating_map_info.map_origin_x
            updating_origin_y = self.agent.updating_map_info.map_origin_y
            
            # 更新机器人位置 - 转换到updating map坐标系
            robot_x = (self.robot_location[0] - updating_origin_x) / self.cell_size
            robot_y = (self.robot_location[1] - updating_origin_y) / self.cell_size
            self.robot_point_updating.set_data([robot_x], [robot_y])
            # 更新朝向箭头
            if hasattr(self.agent, 'heading_theta'):
                dx = np.cos(self.agent.heading_theta)
                dy = np.sin(self.agent.heading_theta)
                self.heading_arrow_updating.set_offsets([[robot_x, robot_y]])
                self.heading_arrow_updating.set_UVC(dx, dy)
            
            # # 清除旧的射线
            # for line in self.ray_lines_updating:
            #     line.remove() if line in self.ax_updating.lines else None
            # self.ray_lines_updating.clear()
            # # 绘制新的射线
            # for ray in self.ray_lines:
            #     if ray:  # 确保射线不为空
            #         ray_x = [point[0] for point in ray]
            #         ray_y = [point[1] for point in ray]
            #         # 绘制射线
            #         line, = self.ax_updating.plot(ray_x, ray_y, 'r-', alpha=0.3, linewidth=0.5, zorder=3)
            #         self.ray_lines_updating.append(line)
                    
            # 更新waypoint位置
            if hasattr(self.agent, 'waypoint') and self.agent.waypoint is not None:
                waypoint_x = (self.agent.waypoint[0] - updating_origin_x) / self.cell_size
                waypoint_y = (self.agent.waypoint[1] - updating_origin_y) / self.cell_size
                self.waypoint_point_updating.set_offsets([[waypoint_x, waypoint_y]])
            else:
                self.waypoint_point_updating.set_offsets(np.array([[]], dtype=float).reshape(0, 2))
            
            # # 更新前沿点
            # if hasattr(self.agent, 'frontier') and len(self.agent.frontier) > 0:
            #     frontier_coords = np.array(list(self.agent.frontier))
            #     frontier_x = (frontier_coords[:, 0] - updating_origin_x) / self.cell_size
            #     frontier_y = (frontier_coords[:, 1] - updating_origin_y) / self.cell_size
            #     frontier_points = np.column_stack((frontier_x, frontier_y))
            #     self.frontier_points_updating.set_offsets(frontier_points)
            #     self.frontier_points_updating.set_visible(True)
            # else:
            #     self.frontier_points_updating.set_offsets(np.array([[]], dtype=float).reshape(0, 2))
                
            self.ax_updating.axis('off')
            
        except Exception as e:
            print(f"渲染updating belief map时出错: {e}")
            # 在发生错误时，不阻止程序继续执行

    def render(self):
        """主渲染函数"""
        if self.render_mode != 'human':
            return
            
        if self.fig is None:
            plt.ion()
            # 创建1行3列的子图，增加一个用于显示updating_belief_map
            self.fig, (self.ax, self.ax_truth, self.ax_updating) = plt.subplots(1, 3, figsize=(15, 5))
            # self.ax.set_title('Global Belief Map')
            # self.ax_truth.set_title('Ground Truth')
            # self.ax_updating.set_title('Updating Belief Map')
        
        # 渲染三个子图
        self._render_belief_map()
        self._render_ground_truth()
        self._render_updating_belief_map()
        
        # 更新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    def plot_env(self, step):
        plt.subplot(1, 3, 1)
        # 使用ground_truth代替robot_belief
        plt.imshow(self.ground_truth, cmap='gray', origin='lower')
        plt.axis('off')
        
        # 绘制机器人和轨迹
        plt.plot((self.robot_location[0] - self.belief_origin_x) / self.cell_size,
                (self.robot_location[1] - self.belief_origin_y) / self.cell_size, 
                'mo', markersize=4, zorder=5)
        plt.plot((np.array(self.trajectory_x) - self.belief_origin_x) / self.cell_size,
                (np.array(self.trajectory_y) - self.belief_origin_y) / self.cell_size, 
                'b', linewidth=2, zorder=1)
        
        # # 绘制动态障碍物 - 现在我们可以显示这些，因为我们展示的是完整地图
        # for obs in self.dynamic_obstacles:
        #     # 转换障碍物位置到栅格坐标
        #     x = (obs['position'][0] - self.belief_origin_x) / self.cell_size
        #     y = (obs['position'][1] - self.belief_origin_y) / self.cell_size
            
        #     # 画出障碍物的圆形范围
        #     circle = plt.Circle((x, y), OBSTACLE_RADIUS * 1.5 / self.cell_size,
        #                         color='red', alpha=0.7, zorder=4)
        #     plt.gca().add_patch(circle)
            
        #     # 画出障碍物的运动路径
        #     path_x = [(obs['waypoint1'][0] - self.belief_origin_x) / self.cell_size,
        #             (obs['waypoint2'][0] - self.belief_origin_x) / self.cell_size]
        #     path_y = [(obs['waypoint1'][1] - self.belief_origin_y) / self.cell_size,
        #             (obs['waypoint2'][1] - self.belief_origin_y) / self.cell_size]
        #     plt.plot(path_x, path_y, 'r--', alpha=0.5, zorder=2)
        
        # 在完整地图上，我们还可以绘制当前的目标点（waypoint）
        if hasattr(self.agent, 'waypoint') and self.agent.waypoint is not None:
            waypoint_x = (self.agent.waypoint[0] - self.belief_origin_x) / self.cell_size
            waypoint_y = (self.agent.waypoint[1] - self.belief_origin_y) / self.cell_size
            plt.scatter(waypoint_x, waypoint_y, c='blue', s=50, marker='*', zorder=5)
        
        # # 我们也可以显示机器人的朝向
        # if hasattr(self.agent, 'heading_theta'):
        #     robot_x = (self.robot_location[0] - self.belief_origin_x) / self.cell_size
        #     robot_y = (self.robot_location[1] - self.belief_origin_y) / self.cell_size
        #     plt.quiver(robot_x, robot_y, 
        #               np.cos(self.agent.heading_theta), np.sin(self.agent.heading_theta),
        #               color='magenta', scale=20, zorder=6)
            
        plt.suptitle('Explored: {:.4g}  Distance: {:.4g}  Collisions: {}  Linear: {:.2f} Angular: {:.2f} Total Reward: {:.2f} Heading Diff: {:.2f} Dis: {:.2f}'.format(
            self.explored_rate, 
            self.travel_dist, 
            self.collision_count,
            self.velocity_command[0],
            self.velocity_command[1],
            self.total_reward,
            self.agent.heading_theta_diff,
            self.distance_to_target
        ))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
        plt.close()
        self.frame_files.append(frame)

    def generate_random_point_in_free_space(self, random_dist):
        """
        在机器人周围RANDOM_DIST范围内的自由空间随机生成一个点
        
        参数:
            random_dist: float, 随机点与机器人的最大距离（米）
        
        返回:
            (x, y): tuple, 随机点的坐标（米）
        """
        random_dist = RANDOM_MAX_DIST
        # 转换距离从米到像素
        random_dist_px = int(random_dist / self.cell_size)
        
        # 获取当前机器人位置（像素坐标）
        robot_x_px = int((self.robot_location[0] - self.belief_origin_x) / self.cell_size)
        robot_y_px = int((self.robot_location[1] - self.belief_origin_y) / self.cell_size)
        
        # 获取地图尺寸
        H, W = self.robot_belief.shape
        
        # 最大尝试次数
        max_attempts = 100
        
        for _ in range(max_attempts):
            # 在圆内随机生成点
            # 随机角度和距离
            angle = np.random.uniform(0, 2*np.pi)
            # 使用sqrt确保点在圆内均匀分布
            distance = np.random.uniform(0, random_dist_px**2)**0.5
            
            # 计算相对偏移
            dx = int(distance * np.cos(angle))
            dy = int(distance * np.sin(angle))
            
            # 计算随机点坐标（像素坐标）
            x_px = robot_x_px + dx
            y_px = robot_y_px + dy
            
            # 检查点是否在地图内
            if 0 <= x_px < W and 0 <= y_px < H:
                # 检查点是否在自由空间 (ROBOT_BELIEF_FREE = 255)
                if self.robot_belief[y_px, x_px] == 255:  # 假设255表示自由空间
                    # 将像素坐标转换回米
                    x_m = x_px * self.cell_size + self.belief_origin_x
                    y_m = y_px * self.cell_size + self.belief_origin_y
                    return (x_m, y_m)
        
        # 如果无法找到自由空间的点，回退到在圆内随机生成点（不考虑自由空间）
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(0, random_dist)
        
        x_m = self.robot_location[0] + distance * np.cos(angle)
        y_m = self.robot_location[1] + distance * np.sin(angle)
        
        print("警告：无法在自由空间找到随机点，生成了一个可能在障碍物内的点")
        return (x_m, y_m)
    
    

class DualStageEnvWrapper(gym.Env):
    def __init__(self, episode_index=0, plot=True, random_waypoint=True, agent=None,render_mode=None):
        
        super(DualStageEnvWrapper, self).__init__()
        self.agent = agent
        self.render_mode = render_mode
        self.env = Env(episode_index, plot, random_waypoint, render_mode)
        self.env.set_agent(self.agent)
        self.env.agent.update_planning_state(self.env.belief_info, self.env.robot_location)
        # 定义 observation_space，假设是一个二维地图 flatten 成 1D
        low = -np.inf * np.ones(4, dtype=np.float32)
        high = np.inf * np.ones(4, dtype=np.float32)
        map_pixels = int(UPDATING_MAP_SIZE / CELL_SIZE)  # 应该是 85
        self.observation_space = gym.spaces.Dict({
            "belief": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(map_pixels, map_pixels, 3),
                dtype=np.float32# 3通道 代表三种状态
            ),
            "robot_state": gym.spaces.Box(
                low=low,
                high=high,
                shape=(4,),
                dtype=np.float32
            )
        })
        # self.observation_space = gym.spaces.Box(
        #     low=np.array(low),
        #     high=np.array(high),
        #     shape=(4,),
        #     dtype=np.float32
        # )
        
        self.action_space = gym.spaces.Box(
            low=np.array([0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )
        self.frame_idx = 0

    def _process_belief_map(self, belief_map):
        # 把belief map 转成三通道
        belief_map = np.stack((belief_map == ROBOT_BELIEF_FREE, belief_map == ROBOT_BELIEF_OCCUPIED, belief_map == ROBOT_BELIEF_UNKNOWN), axis=-1)
        return belief_map
    
    def seed(self, seed=None):
        # 兼容旧版 Gym
        self.reset(seed=seed)
        return [seed]
    
    def _process_robot_state(self, robot_state):
        return robot_state
    
    def _process_obs(self, robot_state, belief_map):
        belief_map_processed = self._process_belief_map(belief_map)
        obs = {
            "belief": belief_map_processed.astype(np.float32),
            "robot_state": np.array(robot_state, dtype=np.float32)
        }
        return obs
    
    def reset(self, seed=None, options=None):
        # 重置环境 随机选一张地图
        robot_state, _ = self.env.reset()
        obs = self._process_obs(robot_state, self.env.agent.updating_map_info.map)
        return obs, {}

    def step(self, action):
        # 用 action 控制机器人移动
        # print("action",action)
        # print("obs", self.env.agent.get_robot_state())
        robot_state, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_obs(robot_state, self.env.agent.updating_map_info.map)
        return obs, reward, terminated, truncated, info

    def render(self):
        # print("render")
        if self.render_mode == 'human':
            self.env.render()  # 你自己内部的渲染逻辑
        pass

    def close(self):
        pass
