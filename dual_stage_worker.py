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

from model_dual_stage import DualStageAgent, WaypointSelector, LocalController, WayPointQNet, ControllerQNetwork
from env import Env
from dual_stage_agent import DualStageAgent
from utils import *
from parameter import *
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Experience = namedtuple('Experience', 
                        ['node_inputs', 'node_padding_mask', 'edge_mask', 'current_index', 
                         'current_edge', 'edge_padding_mask', 'waypoint_idx', 'reward', 
                         'next_node_inputs', 'next_node_padding_mask', 'next_edge_mask', 
                         'next_current_index', 'next_current_edge', 'next_edge_padding_mask',
                         'done', 'robot_state', 'selected_waypoint', 'velocity'])





class DualStageWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        
        self.env = Env(global_step, plot=save_image)
        self.robot = DualStageAgent(policy_net, device=self.device)
        self.robot.env = self.env
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
        next_node_inputs, next_node_padding_mask, next_edge_mask, next_current_index, next_current_edge, next_edge_padding_mask = observation
        self.episode_buffer['next_node_inputs'].append(next_node_inputs)
        self.episode_buffer['next_node_padding_mask'].append(next_node_padding_mask.bool())
        self.episode_buffer['next_edge_mask'].append(next_edge_mask.bool())
        self.episode_buffer['next_current_index'].append(next_current_index)
        self.episode_buffer['next_current_edge'].append(next_current_edge)
        self.episode_buffer['next_edge_padding_mask'].append(next_edge_padding_mask.bool())
        
    def run_episode(self):
        done = False
        need_decision = True
        simulation_time = 0.0
        self.env.set_agent(self.robot)
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        

        self.robot.plot_env()
        self.env.plot_env(0)
            
        max_simulation_time = MAX_EPISODE_TIME
        step_count = 0
        
        while simulation_time < max_simulation_time and step_count < MAX_EPISODE_STEP and not done:
            