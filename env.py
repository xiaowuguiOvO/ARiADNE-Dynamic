import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy

from sensor import sensor_work
from utils import *


class Env:
    def __init__(self, episode_index, plot=False):
        self.episode_index = episode_index
        self.plot = plot
        self.ground_truth, self.robot_cell = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.cell_size = CELL_SIZE  # meter

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
                    
                    # 在第一个点附近找第二个点(限制距离范围,使路径合理)
                    nearby_cells = []
                    min_dist = 10.0 / self.cell_size  # 增加最小距离到10米
                    max_dist = 15.0 / self.cell_size  # 增加最大距离到15米
                    
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
                    speed = MAX_OBSTACLE_SPEED
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

    def calculate_reward(self, dist, collision=False):
        reward = 0
        # 原有的距离惩罚
        reward -= dist / UPDATING_MAP_SIZE * 5
        
        # 碰撞惩罚
        if collision:
            reward -= COLLISION_PENALTY
        else:
            # 接近障碍物惩罚与避障奖励
            min_dist = float('inf')
            for obs in self.dynamic_obstacles:
                dist_to_obs = np.linalg.norm(self.robot_location - obs['position'])
                min_dist = min(min_dist, dist_to_obs)
            
            # 如果太接近障碍物，给予惩罚
            if min_dist < SAFE_DISTANCE:
                reward -= PROXIMITY_PENALTY * (1.0 - min_dist/SAFE_DISTANCE)
            elif min_dist < SAFE_DISTANCE * 1.5:
                # 成功保持安全距离但有风险，小奖励
                reward += 1.0
        
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
    
    def step(self, next_waypoint):
        dist = np.linalg.norm(self.robot_location - next_waypoint)
        self.update_robot_location(next_waypoint)
        self.update_robot_belief()

        # 更新动态障碍物
        self.update_dynamic_obstacles()
        
        # 检查碰撞
        collision = self.check_collision()
        
        self.travel_dist += dist
        self.evaluate_exploration_rate()

        # 将碰撞信息传递给奖励计算函数
        reward = self.calculate_reward(dist, collision)

        return reward

    # def plot_env(self, step):

    #     plt.subplot(1, 3, 1)
    #     plt.imshow(self.robot_belief, cmap='gray')
    #     plt.axis('off')
    #     plt.plot((self.robot_location[0] - self.belief_origin_x) / self.cell_size,
    #              (self.robot_location[1] - self.belief_origin_y) / self.cell_size, 'mo', markersize=4, zorder=5)
    #     plt.plot((np.array(self.trajectory_x) - self.belief_origin_x) / self.cell_size,
    #              (np.array(self.trajectory_y) - self.belief_origin_y) / self.cell_size, 'b', linewidth=2, zorder=1)
    #     plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.explored_rate, self.travel_dist))
    #     plt.tight_layout()
    #     # plt.show()
    #     plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step), dpi=150)
    #     frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
    #     plt.close()
    #     self.frame_files.append(frame)
    def plot_env(self, step):
        plt.subplot(1, 3, 1)
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis('off')
        
        # 绘制机器人和轨迹
        plt.plot((self.robot_location[0] - self.belief_origin_x) / self.cell_size,
                (self.robot_location[1] - self.belief_origin_y) / self.cell_size, 
                'mo', markersize=4, zorder=5)
        plt.plot((np.array(self.trajectory_x) - self.belief_origin_x) / self.cell_size,
                (np.array(self.trajectory_y) - self.belief_origin_y) / self.cell_size, 
                'b', linewidth=2, zorder=1)
        
        # 绘制动态障碍物 - 使用绿色而不是红色
        for obs in self.dynamic_obstacles:
            # 转换障碍物位置到栅格坐标
            x = (obs['position'][0] - self.belief_origin_x) / self.cell_size
            y = (obs['position'][1] - self.belief_origin_y) / self.cell_size
            
            # 画出障碍物的圆形范围 - 改为绿色
            circle = plt.Circle((x, y), OBSTACLE_RADIUS/self.cell_size, 
                            color='green', alpha=0.5, zorder=4)
            plt.gca().add_patch(circle)
            
            # 画出障碍物的运动路径 - 改为绿色虚线
            path_x = [(obs['waypoint1'][0] - self.belief_origin_x) / self.cell_size,
                    (obs['waypoint2'][0] - self.belief_origin_x) / self.cell_size]
            path_y = [(obs['waypoint1'][1] - self.belief_origin_y) / self.cell_size,
                    (obs['waypoint2'][1] - self.belief_origin_y) / self.cell_size]
            plt.plot(path_x, path_y, 'g--', alpha=0.3, zorder=2)  # 绿色虚线表示运动路径
            
            # 画出运动方向箭头 - 改为绿色
            # arrow_length = 2.0  # 箭头长度缩放因子
            # dx = obs['velocity'][0] * arrow_length / self.cell_size
            # dy = obs['velocity'][1] * arrow_length / self.cell_size
            # plt.arrow(x, y, dx, dy, 
            #         head_width=0.3, head_length=0.5, 
            #         fc='green', ec='green', alpha=0.7, zorder=4)
        
        # 更新标题，添加碰撞计数
        plt.suptitle('Explored: {:.4g}  Distance: {:.4g}  Collisions: {}'.format(
            self.explored_rate, self.travel_dist, self.collision_count))
        
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
        plt.close()
        self.frame_files.append(frame)
