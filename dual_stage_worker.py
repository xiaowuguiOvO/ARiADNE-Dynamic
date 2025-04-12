import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import deque, namedtuple
import random
import time
from copy import deepcopy

from dual_stage_model import WaypointSelector, LocalController, WayPointQNet, ControllerQNetwork
from dual_stage_env import Env  # 从正确的文件导入Env类
from dual_stage_agent import DualStageAgent
from utils import *
from parameter import *
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Experience = namedtuple('Experience',
                        ['node_inputs', 'node_padding_mask', 'edge_mask', 'current_index',
                         'current_edge', 'edge_padding_mask', 'waypoint_idx', 'reward',
                         'next_node_inputs', 'next_node_padding_mask', 'next_edge_mask',
                         'next_current_index', 'next_current_edge', 'next_edge_padding_mask',
                         'done', 'robot_state', 'selected_waypoint', 'velocity'])





class DualStageWorker:
    def __init__(self, meta_agent_id, global_step, device='cpu', save_image=False, random_wapoint=False, train_local_controller=True):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.train_local_controller = True
        self.env = Env(global_step, plot=save_image, random_wapoint=random_wapoint)
        self.robot = DualStageAgent(device=self.device, LOAD_LOCAL_CONTROLLER=True)
        self.robot.env = self.env
        self.waypoint_index = None
        # 使用字典替代纯索引列表，提高可读性
        self.episode_buffer = {
            'node_inputs': [],
            'node_padding_mask': [],
            'edge_mask': [],
            'current_index': [],
            'current_edge': [],
            'edge_padding_mask': [],
            'velocity': [],
            'reward': [],
            'done': [],
            'next_node_inputs': [],
            'next_node_padding_mask': [],
            'next_edge_mask': [],
            'next_current_index': [],
            'next_current_edge': [],
            'next_edge_padding_mask': []
        }

    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer['node_inputs'].append(node_inputs)
        self.episode_buffer['node_padding_mask'].append(node_padding_mask.bool())
        self.episode_buffer['edge_mask'].append(edge_mask.bool())
        self.episode_buffer['current_index'].append(current_index)
        self.episode_buffer['current_edge'].append(current_edge)
        self.episode_buffer['edge_padding_mask'].append(edge_padding_mask.bool())

    def save_velocity(self, velocity):
        self.episode_buffer['velocity'].append(velocity)

    def save_reward_done(self, reward, done):
        self.episode_buffer['reward'].append(reward)
        self.episode_buffer['done'].append(done)

    def save_next_observations(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9] += node_inputs
        self.episode_buffer[10] += node_padding_mask.bool()
        self.episode_buffer[11] += edge_mask.bool()
        self.episode_buffer[12] += current_index
        self.episode_buffer[13] += current_edge
        self.episode_buffer[14] += edge_padding_mask.bool()
    
    def run_episode(self):
        done = False
        need_decision = True
        simulation_time = 0.0
        self.env.set_agent(self.robot)
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        self.frams = []
        max_simulation_time = MAX_EPISODE_TIME
        step_count = 0
        
        self.robot.plot_env()
        self.env.plot_env(step_count)
        while simulation_time < max_simulation_time and step_count < MAX_EPISODE_STEP and not done:
            reward, collision = self.env.step()
            # print(f"reward: {reward}, collision: {collision}, need_decision: {need_decision}")
            observation = self.robot.get_observation()
            self.save_observation(observation)
            self.robot.update_planning_state_use_nearest_node(self.env.belief_info, self.env.robot_location)
            # select next waypoint
            next_waypoint = None
            if need_decision:
                if self.env.random_wapoint:
                    success, next_waypoint = self.env.generate_random_waypoint()
                else:
                    next_waypoint, action_index = self.robot.select_next_waypoint(observation)
                    
                need_decision = False
                self.robot.update_waypoint(next_waypoint)
                    # self.robot.decompose_path_to_waypoint()
                self.robot.current_path_index = 0
                print(f"path_points: {self.robot.path_points}")
                # self.robot.update_waypoint([4, -4])

            velocity, state = self.robot.cal_next_velocity(self.robot.path_points[self.robot.current_path_index])
            
            self.robot.update_robot_state(state[0], state[1], state[2], state[3])
            # velocity = [1, 1]
            self.robot.update_velocity(velocity)
            # check arrive waypoint
            if self.robot.check_arrive_waypoint(self.robot.waypoint):
                need_decision = True
                # print("arrive waypoint, need decision")
            # if self.robot.check_arrive_waypoint(self.robot.path_points[self.robot.current_path_index]):
            #     self.robot.current_path_index += 1
            
            if save_exp and not self.train_local_controller:
                self.save_observation(observation)
                self.save_action(action_index)
                next_observation = self.robot.get_observation()
                self.save_next_observations(next_observation)
                self.save_reward_done(reward, done)
                save_exp = False
            # 可视化
            step_count += 1
            if self.save_image:
                self.env.plot_env(step_count)
                self.robot.plot_env(self.robot.next_waypoint_index)

            
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = 1 if done and not collision else 0
        self.perf_metrics['collision_count'] = self.env.collision_count
        self.perf_metrics['simulation_time'] = simulation_time
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)
if __name__ == "__main__":
    torch.manual_seed(4777)
    np.random.seed(4777)
    # model = (NODE_INPUT_DIM, EMBEDDING_DIM)
    # checkpoint = torch.load(model_path + '/checkpoint.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['policy_model'])
    worker = DualStageWorker(0, 22, save_image=True, random_wapoint=True, train_local_controller=True)
    worker.run_episode()
