import torch
import torch.nn as nn
import torch.nn.functional as F
from dual_stage_model import WaypointSelector, LocalController
from parameter import *
from node_manager import NodeManager
import numpy as np
from utils import *
class DualStageAgent:
    def __init__(self, device='cpu'):
        self.device = device
        # self.waypoint_selector = WaypointSelector(device=device)  # 注释掉WaypointSelector初始化
        self.local_controller = LocalController()

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
        
        # 自身速度
        # self.velocity = np.array([0.0, 0.0])  # 当前速度        
        self.nearest_node = None
        self.nearest_node_index = float('inf')
        
        # robot state
        self.v_linear = 0.0
        self.v_angular = 0.0
        self.distance_to_target = 0.0
        self.heading_theta = 0.0
        self.heading_theta_diff = 0.0
        
    def get_robot_state(self):
        return [self.distance_to_target, self.heading_theta_diff, self.v_linear, self.v_angular]
    
    def update_robot_state(self, distance_to_target, heading_theta_diff, v_linear, v_angular):
        self.distance_to_target = distance_to_target
        self.heading_theta_diff = heading_theta_diff
        self.v_linear = v_linear
        self.v_angular = v_angular
        
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
        """简单的更新前沿点方法，仅用于测试"""
        # 在测试LocalController时不需要实际的前沿点，设置为空
        self.frontier = set()
        return
    
    def update_planning_state(self, global_map_info, location):
        self.update_map(global_map_info)
        self.update_updating_map(location)
        self.update_location(location)
        self.update_nearest_node()
        nearest_node_location = np.array([self.nearest_node.x, self.nearest_node.y])
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
        waypoint_logp = self.waypoint_selector(node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask)
        return waypoint_logp
    