import gymnasium as gym  # SB3 v2.4.1使用gymnasium而不是旧的gym
import numpy as np
from local_planner_env import LocalPlannerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os
import stable_baselines3
from sb3_local_env import Env
from dual_stage_agent import DualStageAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from parameter import *
import imageio
from belief_cnn import BeliefFeatureExtractor
import argparse

print(f"SB3版本: {stable_baselines3.__version__}")

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
        
        self.action_space = gym.spaces.Box(
            low=np.array([0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )
        self.frame_idx = 0

    def _process_belief_map(self, belief_map):
        # 把belief map 转成三通道
        belief_map = np.stack((belief_map == ROBOT_BELIEF_FREE, belief_map == ROBOT_BELIEF_OCCUPIED, belief_map == ROBOT_BELIEF_UNKNOWN), axis=-1)
        return belief_map
    
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

def train_with_sb3():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', type=str, help='model path',default=None)
    parser.add_argument('--total-timesteps', type=int, default=100000, help='total step')
    args = parser.parse_args()
    
    # 构造包装后的环境，训练环境可以不开启渲染，评估环境开启渲染有助于观察训练效果
    agent = DualStageAgent(LOAD_LOCAL_CONTROLLER=False)
    env = DualStageEnvWrapper(episode_index=0, plot=False, random_waypoint=True, agent=agent, render_mode=None)
    eval_env = DummyVecEnv([lambda: DualStageEnvWrapper(episode_index=0, plot=False, random_waypoint=True, agent=agent, render_mode='human')])
    policy_kwargs = dict(
        features_extractor_class=BeliefFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128)
    )
    if args.load_model:
        print(f"Loading existing model from {args.load_model}")
        model = PPO.load(args.load_model, env=env,policy_kwargs = policy_kwargs, tensorboard_log="./ppo_sb3_tensorboard", learning_rate=3e-4, n_steps=512, batch_size=128, n_epochs=10, gamma=0.96, gae_lambda=0.95, clip_range=0.2,ent_coef=0.01)
    else:
        model = PPO("MultiInputPolicy", env,policy_kwargs = policy_kwargs, verbose=1, tensorboard_log="./ppo_sb3_tensorboard",
                learning_rate=1e-3, n_steps=512, batch_size=128, n_epochs=10, gamma=0.96, gae_lambda=0.95, clip_range=0.2,ent_coef=0.02)
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_sb3_tensorboard",
    #             learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,ent_coef=0.01)
    # 回调函数保存检查点和定期评估
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./ppo_checkpoints/',
                                             name_prefix='ppo_model')

    eval_callback = EvalCallback(eval_env, best_model_save_path='./ppo_best_model/',
                                 log_path='./ppo_eval_logs/', eval_freq=2000    ,
                                 deterministic=True, render=False, n_eval_episodes=1)
    # 训练
    total_timesteps = 1000000  # 
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    # 保存最终模型
    model.save("ppo_sb3_final_model")
    print("训练结束，模型已保存。")

if __name__ == "__main__":
    train_with_sb3()
