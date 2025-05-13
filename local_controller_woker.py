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
from local_controller_env import Env  # 从正确的文件导入Env类
from dual_stage_agent import DualStageAgent
from utils import *
from parameter import *
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LocalControllerWorker:
    def __init__(self, meta_agent_id, global_step, device='cpu', save_image=False, random_wapoint=False, train_local_controller=True):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.train_local_controller = True
        self.env = Env(global_step, plot=save_image, random_wapoint=random_wapoint)
        self.robot = DualStageAgent(device=self.device, LOAD_LOCAL_CONTROLLER=False)
        self.robot.env = self.env
        self.waypoint_index = None
        self.perf_metrics = dict()
        self.episode_buffer = []
        for i in range(5):
            self.episode_buffer.append([])
        # 添加噪声衰减相关参数
        self.initial_exploration_noise = 0.3  # 初始噪声大小
        self.exploration_noise = self.initial_exploration_noise  # 当前噪声大小
        self.min_exploration_noise = 0.05  # 最小噪声大小
        self.exploration_decay = 0.995  # 衰减率，每次执行衰减时乘以这个值
        self.exploration_decay_steps = 100  # 每多少步进行一次衰减
        self.steps_done = 0  # 已完成的步数
    def save_robot_state(self, state):
        self.episode_buffer[0].append(torch.tensor(state, dtype=torch.float32))
    
    def save_next_robot_state(self, state):
        self.episode_buffer[1].append(torch.tensor(state, dtype=torch.float32))
    
    def save_action(self, action):
        self.episode_buffer[2].append(torch.tensor(action, dtype=torch.float32))

    def save_reward_done(self, reward, done):
        self.episode_buffer[3].append(torch.tensor(reward, dtype=torch.float32))
        self.episode_buffer[4].append(torch.tensor(done, dtype=torch.float32))
    
    def run_episode(self):
        done = False
        need_decision = True
        simulation_time = 0.0
        self.env.set_agent(self.robot)
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        self.frams = []
        max_simulation_time = MAX_EPISODE_TIME
        step_count = 0
        if self.save_image:
            self.robot.plot_env()
            self.env.plot_env(step_count)
            
        while not done:

            reward, dynamic_collision, wall_collision, done = self.env.step()
            if step_count > MAX_EPISODE_STEP:
                done = True
            observation = self.robot.get_observation()
            action_index = None
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

            self.save_robot_state(self.robot.get_robot_state())
            self.save_action([self.robot.v_linear, self.robot.v_angular])
            velocity, state = self.robot.cal_next_velocity(self.robot.waypoint)
            
            # 添加基于当前噪声水平的探索噪声
            noise = np.random.normal(0, self.exploration_noise, size=velocity.shape)
            # print(f"noise: {noise}")
            velocity = velocity + noise
            velocity[0] = max(0, min(velocity[0], MAX_LINEAR_VELOCITY))
            velocity[1] = max(-MAX_ANGULAR_VELOCITY, min(velocity[1], MAX_ANGULAR_VELOCITY))
            # 更新步数计数器并衰减噪声
            self.steps_done += 1
            # print(f"steps_done: {self.steps_done}")
            # print(self.exploration_noise)
            if self.steps_done % self.exploration_decay_steps == 0:
                self.exploration_noise = max(
                    self.min_exploration_noise, 
                    self.exploration_noise * self.exploration_decay
                )
                print(f"[Episode {self.global_step}] Exploration noise decayed to {self.exploration_noise:.4f}")
            self.robot.update_robot_state(state[0], state[1], state[2], state[3])
            self.save_next_robot_state(state)
            self.save_reward_done(reward, done)
            self.robot.update_velocity(velocity)
            # check arrive waypoint
            if self.robot.check_arrive_waypoint(self.robot.waypoint):
                need_decision = True
                # print("arrive waypoint, need decision")
            # if self.robot.check_arrive_waypoint(self.robot.path_points[self.robot.current_path_index]):
            #     self.robot.current_path_index += 1
            # 可视化
            step_count += 1
            if self.save_image:
                self.env.plot_env(step_count)
                self.robot.plot_env(self.robot.next_waypoint_index)

            
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = 1 if done else 0
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
    worker = LocalControllerWorker(0, 22, save_image=True, random_wapoint=True, train_local_controller=True)
    worker.run_episode()
