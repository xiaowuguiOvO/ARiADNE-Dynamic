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
from parameter import *
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
        self.perf_metrics = dict()
        
        self.episode_buffer = []
        for i in range(15):
            self.episode_buffer.append([])

    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index
        
    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

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
        save_exp = True
        self.robot.plot_env()
        if self.save_image:
            self.env.plot_env(step_count)
        next_waypoint = None
        while simulation_time < max_simulation_time and step_count < MAX_EPISODE_STEP and not done:
            reward, collision = self.env.step()
            # print(f"reward: {reward}, collision: {collision}, need_decision: {need_decision}")
            observation = self.robot.get_observation()
            self.robot.update_planning_state_use_nearest_node(self.env.belief_info, self.env.robot_location)
            # select next waypoint
            
            if need_decision:
                if self.env.random_wapoint:
                    success, next_waypoint = self.env.generate_random_waypoint()
                else:
                    next_waypoint, action_index = self.robot.select_next_waypoint(observation)
                    
                need_decision = False
                save_exp = True
                self.robot.update_waypoint(next_waypoint)
                self.robot.current_path_index = 0
            # update velocity
            self.robot.update_robot_state(next_waypoint)
            state = self.robot.get_robot_state()
            robot_local_belief = self.robot.get_robot_local_belief()
            action = self.robot.get_local_action(state, robot_local_belief)
            self.robot.update_velocity(action)
            # check arrive waypoint
            if self.robot.check_arrive_waypoint(self.robot.waypoint):
                need_decision = True
                # print("arrive waypoint, need decision")
            # if self.robot.check_arrive_waypoint(self.robot.path_points[self.robot.current_path_index]):
            #     self.robot.current_path_index += 1
            
            if save_exp:
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
    worker = DualStageWorker(0, 22, save_image=True, random_wapoint=False, train_local_controller=False)
    worker.run_episode()
