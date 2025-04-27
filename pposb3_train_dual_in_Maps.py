import gymnasium as gym  # SB3 v2.4.1使用gymnasium而不是旧的gym
import numpy as np
from local_planner_env import LocalPlannerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os
import stable_baselines3
from sb3_dual_env import Env
from dual_stage_agent import DualStageAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from parameter import *
import imageio
from belief_cnn import BeliefFeatureExtractor
import argparse
print(f"SB3版本: {stable_baselines3.__version__}")
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_sac_model import WaypointSelectorSAC
from discrete_sac import DiscreteSAC, DiscreteSACPolicy

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.waypoint_selector = WaypointSelectorSAC(
            node_dim=4,
            embedding_dim=128,
            action_dim=features_dim
        )
    
    def forward(self, observations):
        # 从observations字典中提取所需的输入
        return self.waypoint_selector(
            observations["node_inputs"],
            observations["node_padding_mask"],
            observations["edge_mask"],
            observations["current_index"],
            observations["current_edge"],
            observations["edge_padding_mask"]
        )

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
        # self.observation_space = gym.spaces.Dict({
        #     "belief": gym.spaces.Box(
        #         low=0.0,
        #         high=1.0,
        #         shape=(map_pixels, map_pixels, 3),
        #         dtype=np.float32# 3通道 代表三种状态
        #     ),
        #     "robot_state": gym.spaces.Box(
        #         low=low,
        #         high=high,
        #         shape=(4,),
        #         dtype=np.float32
        #     )
        # })
        # self.action_space = gym.spaces.Box(
        #     low=np.array([0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        # )
        self.observation_space = gym.spaces.Dict({
            "node_inputs": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, NODE_PADDING_SIZE, 4),  # node_coords(2) + utility(1) + guidepost(1)
                dtype=np.float32
            ),
            "node_padding_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(1, 1, NODE_PADDING_SIZE),
                dtype=np.int16
            ),
            "edge_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(1, NODE_PADDING_SIZE, NODE_PADDING_SIZE),
                dtype=np.float32
            ),
            "current_index": gym.spaces.Box(
                low=0,
                high=NODE_PADDING_SIZE,
                shape=(1, 1, 1),
                dtype=np.int64
            ),
            "current_edge": gym.spaces.Box(
                low=0,
                high=NODE_PADDING_SIZE,
                shape=(1, K_SIZE, 1),
                dtype=np.int64
            ),
            "edge_padding_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(1, 1, K_SIZE),
                dtype=np.int16
            )
        })
        
        self.action_space = gym.spaces.Discrete(NODE_PADDING_SIZE)
        
    def _process_belief_map(self, belief_map):
        # 把belief map 转成三通道
        belief_map = np.stack((belief_map == ROBOT_BELIEF_FREE, belief_map == ROBOT_BELIEF_OCCUPIED, belief_map == ROBOT_BELIEF_UNKNOWN), axis=-1)
        return belief_map
    
    def _process_robot_state(self, robot_state):
        return robot_state
    
    def _process_obs(self):
        obs = obs
        return obs
    
    def reset(self, seed=None, options=None):
        # 重置环境 随机选一张地图
        robot_state, _ = self.env.reset()
        obs = self._process_obs()
        return obs, {}

    def step(self, action):
        # 用 action 控制机器人移动
        # print("action",action)
        # print("obs", self.env.agent.get_robot_state())
        # action = [1, 0]
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_obs(obs, self.env.agent.updating_map_info.map)
        return obs, reward, terminated, truncated, info

    def render(self):
        # print("render")
        if self.render_mode == 'human':
            self.env.render()  # 你自己内部的渲染逻辑
        pass

    def close(self):
        pass

def train_with_discrete_sac():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', type=str, help='model path', default=None)
    parser.add_argument('--total-timesteps', type=int, default=100000, help='total step')
    args = parser.parse_args()
    
    # 构造环境
    agent = DualStageAgent(LOAD_LOCAL_CONTROLLER=True)
    env = DualStageEnvWrapper(episode_index=0, plot=False, random_waypoint=True, 
                             agent=agent, render_mode=None)
    eval_env = DummyVecEnv([lambda: DualStageEnvWrapper(
        episode_index=0, plot=False, random_waypoint=True, 
        agent=agent, render_mode='human')])
    
    # 策略配置
    policy_kwargs = dict(
        features_extractor_class=CustomExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    )
    
    # 创建或加载模型
    if args.load_model:
        model = DiscreteSAC.load(args.load_model, env=env, tensorboard_log="./discrete_sac_tensorboard/")
    else:
        model = DiscreteSAC(
            DiscreteSACPolicy,
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=100,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            tensorboard_log="./discrete_sac_tensorboard/"
        )
    
    # 回调函数保持不变
    checkpoint_callback = CheckpointCallback(...)
    eval_callback = EvalCallback(...)
    
    # 训练
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # 保存最终模型
    model.save("discrete_sac_final_model")

if __name__ == "__main__":
    train_with_discrete_sac()