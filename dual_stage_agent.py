import torch
import torch.nn as nn
import torch.nn.functional as F
from dual_stage_model import WaypointSelector, LocalController
from parameter import *
from node_manager import NodeManager
import numpy as np
from utils import *
import matplotlib.pyplot as plt
class DualStageAgent:
    def __init__(self, device='cpu', LOAD_LOCAL_CONTROLLER=False):
        self.device = device
        self.waypoint_selector = WaypointSelector(node_dim=NODE_INPUT_DIM, embedding_dim=EMBEDDING_DIM)  
        self.local_controller = LocalController()
        self.LOAD_LOCAL_CONTROLLER = LOAD_LOCAL_CONTROLLER
        if self.LOAD_LOCAL_CONTROLLER:
            self._load_local_controller()
        self.location = None
        self.map_info = None
        
        # map related parameters
        self.cell_size = CELL_SIZE
        self.node_resolution = NODE_RESOLUTION 
        self.updating_map_size = UPDATING_MAP_SIZE
        self.updating_map_info = None
        
        # frontiers
        self.frontier = set()
        
        # node managers
        # 定义一个空的plot函数，用于替代
        self.plot = lambda *args, **kwargs: None
        self.node_manager = NodeManager(plot=self.plot)
        # graph
        self.node_coords, self.utility, self.guidepost = None, None, None
        self.adjacent_matrix, self.neighbor_indices = None, None
        self.next_waypoint_index = None
        # 自身速度
        # self.velocity = np.array([0.0, 0.0])  # 当前速度        
        self.nearest_node = None
        self.nearest_node_index = float('inf')
        
        self.waypoint = [0, 0]
        self.path_points = []
        self.current_path_index = 0
        
        self.use_path_following = True
        
        # robot state
        self.v_linear = 0.0
        self.v_angular = 0.0
        self.distance_to_target = 0.0
        self.heading_theta = 0.0
        self.heading_theta_diff = 0.0
    
    def update_waypoint(self, waypoint):
        self.waypoint = waypoint
    
    def _load_local_controller(self):
        # 先创建模型实例
        self.local_controller = LocalController(state_dim=4, action_dim=2).to(self.device)
        # 然后加载状态字典
        state_dict = torch.load(LOCAL_CONTROLLER_PATH, map_location=self.device)
        self.local_controller.load_state_dict(state_dict)
        self.local_controller.eval()  # 设置为评估模式
    
    def get_robot_state(self):
        return [self.distance_to_target, self.heading_theta_diff, self.v_linear, self.v_angular]
    
    def update_robot_state(self, distance_to_target, heading_theta_diff, v_linear, v_angular):
        self.distance_to_target = distance_to_target
        self.heading_theta_diff = heading_theta_diff
        self.v_linear = v_linear
        self.v_angular = v_angular
    
    def check_arrive_waypoint(self, waypoint):
        # print(f"waypoint distance: {self.cal_dist_to_waypoint(waypoint)}")
        if self.cal_dist_to_waypoint(waypoint) < WAYPOINT_THRESHOLD:
            return True
        else:
            return False
    
    def update_map(self, map_info):
        self.map_info = map_info
    
    def update_updating_map(self, location):
        self.updating_map_info = self.get_updating_map(location)
    
    def update_location(self, location):
        self.location = location

    def get_updating_map(self, location):
        # the map includes all nodes that may be updating
        updating_map_origin_x = (location[
                                  0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[
                                  1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = (self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1))
        max_y = (self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1))

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        updating_map = self.map_info.map[
                    updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1]+1,
                    updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0]+1]

        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info
    
    def update_nearest_node(self):
        """更新与当前位置最接近的节点"""
        if self.location is None:
            return False
        # 检查四叉树是否为空
        if len(self.node_manager.nodes_dict) == 0:
            print("警告：四叉树为空，无法找到最近节点")
            return False
        try:
            # 使用四叉树的nearest_neighbors方法
            nearest_node = self.node_manager.nodes_dict.nearest_neighbors(
                self.location.tolist(), count=1)
            if nearest_node and len(nearest_node) > 0:
                # 更新最近节点
                self.nearest_node = nearest_node[0]
                # 计算距离
                node_pos = np.array([self.nearest_node.x, self.nearest_node.y])
                self.nearest_node_distance = np.linalg.norm(self.location - node_pos)
                # 可选：标记节点为已访问
                if hasattr(self.nearest_node, 'data') and self.nearest_node.data is not None:
                    self.nearest_node.data.set_visited()
                return True
            else:
                print("未找到任何最近节点")
        except Exception as e:
            print(f"更新最近节点时出错: {e}")
        return False
    
    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)
    

    def update_planning_state(self, global_map_info, location):
        self.update_map(global_map_info)
        self.update_location(location)
        # self.location = location
        # self.update_nearest_node()
        self.update_updating_map(self.location)
        self.update_frontiers()
        self.node_manager.update_graph(self.location,
                                       self.frontier,
                                       self.updating_map_info,
                                       self.map_info)
        node = self.node_manager.nodes_dict.find(location.tolist())
        if node is not None:
            node.data.set_visited()
        self.node_coords, self.utility, self.guidepost, self.adjacent_matrix, self.current_index, self.neighbor_indices, self.obstacle_velocities = \
            self.update_observation()
    def update_planning_state_use_nearest_node(self, global_map_info, location):
        self.update_map(global_map_info)
        # self.update_location(location)
        self.location = location
        self.update_nearest_node()
        nearest_node_location = np.array([self.nearest_node.x, self.nearest_node.y])
        self.update_updating_map(nearest_node_location)
        self.update_frontiers()
        self.node_manager.update_graph(nearest_node_location,
                                       self.frontier,
                                       self.updating_map_info,
                                       self.map_info)
        self.node_coords, self.utility, self.guidepost, self.adjacent_matrix, self.current_index, self.neighbor_indices, self.obstacle_velocities = \
            self.update_observation()

    def update_observation(self):
        all_node_coords = []
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        guidepost = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        obstacle_velocities = np.zeros((n_nodes, 2))  # 默认值为 (0, 0)

        for i, coords in enumerate(all_node_coords):
            node = self.node_manager.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            guidepost.append(node.visited)
            for neighbor in node.neighbor_set:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                assert index is not None
                index = index[0][0]
                adjacent_matrix[i, index] = 0
            
        # 遍历动态障碍物
        if hasattr(self, 'env') and hasattr(self.env, 'dynamic_obstacles'):
            for obs in self.env.dynamic_obstacles:
                obs_pos = obs['position']
                obs_vel = obs['velocity']
                # 计算障碍物与智能体的距离
                distance_to_agent = np.linalg.norm(self.location - obs_pos)
                # 找到最近的节点
                distances_to_nodes = np.linalg.norm(all_node_coords - obs_pos, axis=1)
                closest_node_index = np.argmin(distances_to_nodes)
                # 检查障碍物是否在 SENSOR 范围内
                if distance_to_agent <= SENSOR_RANGE:
                    obstacle_velocities[closest_node_index] = obs_vel


        utility = np.array(utility)
        guidepost = np.array(guidepost)

        # current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        if self.nearest_node is not None:
            current_index = np.argwhere(node_coords_to_check == self.nearest_node.x + self.nearest_node.y * 1j)[0][0]
        else:
            current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        return all_node_coords, utility, guidepost, adjacent_matrix, current_index, neighbor_indices, obstacle_velocities

    def get_observation(self):
        node_coords = self.node_coords
        node_utility = self.utility.reshape(-1, 1)
        node_guidepost = self.guidepost.reshape(-1, 1)
        current_index = self.current_index
        edge_mask = self.adjacent_matrix
        current_edge = self.neighbor_indices
        n_node = node_coords.shape[0]

        current_node_coords = node_coords[self.current_index]
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                            node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                           axis=-1) / UPDATING_MAP_SIZE
        node_utility = node_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        # 障碍物速度
        obstacle_velocities = self.obstacle_velocities
        # node_inputs = np.concatenate((node_coords, node_utility, node_guidepost, obstacle_velocities), axis=1)
        node_inputs = np.concatenate((node_coords, node_utility, node_guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        assert node_coords.shape[0] < NODE_PADDING_SIZE, print(node_coords.shape[0], NODE_PADDING_SIZE)
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        padding = torch.nn.ConstantPad2d(
            (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(current_edge == self.current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]

    def select_next_waypoint(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        with torch.no_grad():
            waypoint_logp = self.waypoint_selector(node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask)
        
        waypoint_index = torch.multinomial(waypoint_logp.exp(), 1).long().squeeze(1)
        next_waypoint = self.node_coords[waypoint_index]
        # self.next_waypoint_index = waypoint_index
        return next_waypoint, waypoint_index
    
    def cal_next_velocity(self, waypoint):
        distance_to_target = self.cal_dist_to_waypoint(waypoint)
        heading_theta_diff = self.cal_heading_theta_diff_to_waypoint(waypoint)
        state = np.array([distance_to_target, 
                         heading_theta_diff,
                         self.v_linear, 
                         self.v_angular])
                
        with torch.no_grad(): 
            velocity = self.local_controller(torch.FloatTensor(state).to(self.device))
        velocity = velocity.cpu().numpy()
        return velocity, state

    def update_velocity(self, velocity):
        self.v_linear = velocity[0]
        self.v_angular = velocity[1]
    def cal_dist_to_waypoint(self, waypoint):
        # print(f"self.location: {self.location}, self.waypoint: {self.waypoint}")
        return np.linalg.norm(self.location - waypoint)

    def cal_heading_theta_to_waypoint(self, waypoint):
        return np.arctan2(waypoint[1] - self.location[1], waypoint[0] - self.location[0])

    def cal_heading_theta_diff_to_waypoint(self, waypoint):
        # 计算指向目标点的绝对方向
        target_direction = np.arctan2(waypoint[1] - self.location[1], 
                                      waypoint[0] - self.location[0])
        # 获取机器人当前朝向（假设已在某处存储）
        current_heading = self.heading_theta  # 或者其他存储当前朝向的变量
        # 计算角度差，并标准化到[-π, π]范围
        self.heading_theta_diff = np.arctan2(np.sin(target_direction - current_heading), 
                                 np.cos(target_direction - current_heading))
        return self.heading_theta_diff

    
    def decompose_path_to_waypoint(self):
        """
        使用A*算法将目标waypoint分解成一系列小的路径点
        """
        # 清空之前的路径
        self.path_points = []
        self.current_path_index = 0
        # 获取当前位置最近的节点作为起点
        if self.location is None:
            print("机器人位置未设置，无法规划路径")
            return
        # 更新最近节点
        self.update_nearest_node()
        if self.nearest_node is None:
            print("找不到起点节点，无法规划路径")
            return
        start_coords = [self.nearest_node.x, self.nearest_node.y]
        # 获取目标点最近的节点
        # 确保 waypoint 是列表格式
        waypoint_list = self.waypoint.tolist() if isinstance(self.waypoint, np.ndarray) else list(self.waypoint)
        nearest_to_waypoint = self.node_manager.nodes_dict.nearest_neighbors(
            waypoint_list, count=1)
        if not nearest_to_waypoint or len(nearest_to_waypoint) == 0:
            print("找不到终点节点，无法规划路径")
            return
        end_node = nearest_to_waypoint[0]
        end_coords = [end_node.x, end_node.y]
        # 使用A*算法计算路径
        path, path_length = self.node_manager.a_star(start_coords, end_coords)
        if not path or len(path) == 0:
            print(f"无法找到从 {start_coords} 到 {end_coords} 的路径")
            # 如果找不到路径，直接使用目标点
            self.path_points = [np.array(self.waypoint)]
        else:
            # self.path_points.append(np.array(start_coords))
            # 添加中间路径点
            for point in path:
                self.path_points.append(np.array(point))
            # 添加终点（确保最后的目标是原始的waypoint而不是最近节点）
            if np.linalg.norm(np.array(self.waypoint) - np.array(end_coords)) > 0.1:
                self.path_points.append(np.array(self.waypoint))
        # 设置当前目标为第一个路径点
        if len(self.path_points) > 0:
            self.current_target = self.path_points[0]
            self.current_path_index = 0
        else:
            self.current_target = np.array(self.waypoint)
        return self.path_points
        
    def plot_env(self, waypoint_index=None):
        plt.switch_backend('agg')
        plt.figure(figsize=(18, 5))
        
        plt.subplot(1, 3, 2)
        nodes = get_cell_position_from_coords(self.node_coords, self.map_info)
        if len(self.frontier) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2)
        robot = get_cell_position_from_coords(self.location, self.map_info)
        plt.imshow(self.map_info.map, cmap='gray', origin='lower')
        plt.axis('off')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.utility, zorder=2)
        for node, utility in zip(nodes, self.utility):
            plt.text(node[0], node[1], str(utility), zorder=3)
        plt.plot(robot[0], robot[1], 'mo', markersize=8, zorder=5)
        # 添加朝向箭头
        plt.quiver(robot[0], robot[1], np.cos(self.heading_theta), np.sin(self.heading_theta), 
                    color='m', scale=32, zorder=5)
            # 绘制分解后的路径
        if self.use_path_following and len(self.path_points) > 0:
            path_points = np.array(self.path_points)
            path_points_cells = (path_points - np.array([self.map_info.map_origin_x, self.map_info.map_origin_y])) / self.cell_size
            # 绘制路径线
            plt.plot(path_points_cells[:, 0], path_points_cells[:, 1], 'g-', linewidth=2, zorder=3)
            # 绘制路径点
            plt.scatter(path_points_cells[:, 0], path_points_cells[:, 1], c='g', s=30, zorder=4)
            # 标记当前目标点
            # current_target = None
            # if self.current_path_index < len(path_points):
            #     current_target = self.path_points[self.current_path_index]
            # if current_target is not None:
            #     current_target_cell = (current_target - np.array([self.map_info.map_origin_x, self.map_info.map_origin_y])) / self.cell_size
            #     plt.scatter(current_target_cell[0], current_target_cell[1], c='c', s=80, marker='*', zorder=6)

        # # 添加动态障碍物到中间子图
        # if hasattr(self, 'env') and hasattr(self.env, 'dynamic_obstacles'):
        #     for obs in self.env.dynamic_obstacles:
        #         # 转换障碍物位置到栅格坐标
        #         cell_x = int((obs['position'][0] - self.map_info.map_origin_x) / self.map_info.cell_size)
        #         cell_y = int((obs['position'][1] - self.map_info.map_origin_y) / self.map_info.cell_size)
        #         # 画出障碍物圆形 - 使用更明显的颜色和更大的尺寸
        #         circle = plt.Circle((cell_x, cell_y), OBSTACLE_RADIUS * 1.5 / self.cell_size,  # 增加半径
        #                             color='red', alpha=0.7, zorder=4)  # 使用红色
        #         plt.gca().add_patch(circle)
        #         # 画出障碍物路径 - 使用红色虚线
        #         path_x = [int((obs['waypoint1'][0] - self.map_info.map_origin_x) / self.map_info.cell_size),
        #                 int((obs['waypoint2'][0] - self.map_info.map_origin_x) / self.map_info.cell_size)]
        #         path_y = [int((obs['waypoint1'][1] - self.map_info.map_origin_y) / self.map_info.cell_size),
        #                 int((obs['waypoint2'][1] - self.map_info.map_origin_y) / self.map_info.cell_size)]
        #         plt.plot(path_x, path_y, 'r--', alpha=0.5, zorder=2)  # 红色虚线表示运动路径
# 画所有节点
        for coords in self.node_coords:
            # 画节点之间的路径
            node = self.node_manager.nodes_dict.find(coords.tolist()).data
            for neighbor_coords in node.neighbor_set:
                end = (np.array(neighbor_coords) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                        (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)

        # 绘制目标点 (waypoint)
        # print(self.waypoint[0], self.waypoint[1])
        plt.scatter((self.waypoint[0] - self.map_info.map_origin_x) / self.cell_size, 
                    (self.waypoint[1] - self.map_info.map_origin_y) / self.cell_size, 
                    c='blue', s=50, zorder=4)

        plt.subplot(1, 3, 3)
        plt.imshow(self.map_info.map, cmap='gray', origin='lower')
        plt.axis('off')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.guidepost, zorder=2)
        plt.scatter((self.waypoint[0] - self.map_info.map_origin_x) / self.cell_size, 
            (self.waypoint[1] - self.map_info.map_origin_y) / self.cell_size, 
            c='blue', s=50, zorder=4)
        plt.plot(robot[0], robot[1], 'mo', markersize=8, zorder=5)
        plt.quiver(robot[0], robot[1], np.cos(self.heading_theta), np.sin(self.heading_theta), 
            color='m', scale=32, zorder=5)
        # 绘制分解后的路径
        # if self.use_path_following and len(self.path_points) > 0:
        #     path_points = np.array(self.path_points)
        #     path_points_cells = (path_points - np.array([self.map_info.map_origin_x, self.map_info.map_origin_y])) / self.cell_size
        #     # 绘制路径线
        #     plt.plot(path_points_cells[:, 0], path_points_cells[:, 1], 'g-', linewidth=2, zorder=3)
        #     # 绘制路径点
        #     plt.scatter(path_points_cells[:, 0], path_points_cells[:, 1], c='g', s=30, zorder=4)
            # 标记当前目标点
            # current_target = None
            # if self.current_path_index < len(path_points):
            #     current_target = self.path_points[self.current_path_index]
            # if current_target is not None:
            #     current_target_cell = (current_target - np.array([self.map_info.map_origin_x, self.map_info.map_origin_y])) / self.cell_size
            #     plt.scatter(current_target_cell[0], current_target_cell[1], c='c', s=80, marker='*', zorder=6)

        # # 添加动态障碍物到右侧子图
        # if hasattr(self, 'env') and hasattr(self.env, 'dynamic_obstacles'):
        #     for obs in self.env.dynamic_obstacles:
        #         # 转换障碍物位置到栅格坐标
        #         cell_x = int((obs['position'][0] - self.map_info.map_origin_x) / self.map_info.cell_size)
        #         cell_y = int((obs['position'][1] - self.map_info.map_origin_y) / self.map_info.cell_size)
                
        #         # 画出障碍物圆形 - 使用更明显的颜色和更大的尺寸
        #         circle = plt.Circle((cell_x, cell_y), OBSTACLE_RADIUS * 1.5 / self.cell_size,  # 增加半径
        #                             color='red', alpha=0.7, zorder=4)  # 使用红色
        #         plt.gca().add_patch(circle)