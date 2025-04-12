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

class Env:
    def __init__(self, episode_index, plot=False, random_wapoint=False):
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
        self.distance_since_last_decision = 0.0
        self.time_since_last_decision = 0.0
        
        self.velocity_command = np.array([0.0, 0.0])
        self.total_reward = 0.0
    
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

    def calculate_reward(self, dist, collision=False, wall_collision=False):
        reward = 0
        # 原有的距离惩罚
        # reward -= dist / UPDATING_MAP_SIZE * 5
        
        # 碰撞惩罚
        if collision:
            reward -= COLLISION_PENALTY
        elif wall_collision:
            reward -= WALL_COLLISION_PENALTY

        # 原有的探索奖励
        global_frontiers = get_frontier_in_map(self.belief_info)
        if len(global_frontiers) == 0:
            delta_num = len(self.global_frontiers)
        else:
            observed_frontiers = self.global_frontiers - global_frontiers
            delta_num = len(observed_frontiers)

        reward += delta_num / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)

        self.global_frontiers = global_frontiers
        self.old_belief = deepcopy(self.robot_belief)

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
        在机器人附近的NodeManager中随机选择一个节点作为目标点
        
        Returns:
            bool: 是否成功生成目标点
            np.array: 生成的目标点坐标，如果失败则为None
        """
        if self.agent is None or self.agent.node_manager is None:
            print("警告：agent或NodeManager未初始化")
            return False, None
            
        # 获取当前位置
        current_location = self.robot_location
        
        # 获取NodeManager中的所有节点
        candidate_nodes = []
        for node in self.agent.node_manager.nodes_dict.__iter__():
            node_pos = np.array([node.x, node.y])
            distance = np.linalg.norm(node_pos - current_location)
            
            # 检查节点是否在指定距离范围内
            if 0.5 <= distance <= RANDOM_DIST:
                candidate_nodes.append(node)
        
        # 如果没有符合条件的节点，返回失败
        if not candidate_nodes:
            print(f"在距离{RANDOM_DIST}米范围内没有找到合适的节点")
            return False, None
    
        # 随机选择一个候选节点
        selected_node = random.choice(candidate_nodes)
        selected_waypoint = np.array([selected_node.x, selected_node.y])
        
        # print(f"从NodeManager中选择随机目标点: {selected_waypoint}, 距离: {np.linalg.norm(selected_waypoint - current_location):.2f}m")
        return True, selected_waypoint
    
    def step(self):
        """
        执行一个模拟步骤，使用agent的当前速度（线速度和角速度）
        返回:
            reward: 奖励值
            collision: 是否发生碰撞
            need_decision: 是否需要做新决策
        """
        done = False
        if self.agent is None:
            raise ValueError("必须先使用set_agent设置代理")
        
        linear_vel = self.agent.v_linear
        angular_vel = self.agent.v_angular
        velocity_command = np.array([linear_vel, angular_vel])
        # print(f"velocity_command: {velocity_command}")
        # 初始化机器人朝向(如果不存在)
        if not hasattr(self, 'robot_orientation'):
            self.robot_orientation = 0.0  # 初始朝向
        
        # 更新朝向
        self.robot_orientation += angular_vel * self.step_size
        # 标准化到 [-π, π]
        self.robot_orientation = np.arctan2(np.sin(self.robot_orientation), np.cos(self.robot_orientation))
        self.agent.heading_theta = self.robot_orientation
        # 使用线速度和朝向计算x,y方向的速度分量
        vx = linear_vel * np.cos(self.robot_orientation)
        vy = linear_vel * np.sin(self.robot_orientation)
        cartesian_velocity = np.array([vx, vy])
        
        # 使用笛卡尔速度计算下一个位置
        current_pos = self.robot_location.copy()
        next_pos = current_pos + cartesian_velocity * self.step_size
        
        wall_collision = self.check_wall_collision(next_pos)
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
        
        # 计算当前步骤移动距离 - 使用笛卡尔速度的模长
        step_distance = np.linalg.norm(cartesian_velocity) * self.step_size
        
        # 更新位置
        old_location = self.robot_location.copy()
        
        # 更新位置
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
        
        # 更新移动距离
        moved_dist = np.linalg.norm(self.robot_location - old_location)
        total_dist += moved_dist
        # 更新动态障碍物位置
        self.update_dynamic_obstacles(self.step_size)
        
        self.update_robot_belief()
        self.agent.belief_info = self.belief_info
        self.agent.update_local_belief_map()
        # 累计总移动距离
        self.travel_dist += total_dist
        # 评估探索率
        self.evaluate_exploration_rate()
        # 计算奖励
        reward = self.calculate_reward(total_dist, dynamic_collision, wall_collision)
        self.total_reward += reward

        if wall_collision:
            done = True
        return reward, dynamic_collision, wall_collision, done
            
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
            
        plt.suptitle('Explored: {:.4g}  Distance: {:.4g}  Collisions: {}  Linear: {:.2f} Angular: {:.2f} Total Reward: {:.2f} Heading: {:.2f}'.format(
            self.explored_rate, 
            self.travel_dist, 
            self.collision_count,
            self.velocity_command[0],
            self.velocity_command[1],
            self.total_reward,
            self.agent.heading_theta
        ))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
        plt.close()
        self.frame_files.append(frame)