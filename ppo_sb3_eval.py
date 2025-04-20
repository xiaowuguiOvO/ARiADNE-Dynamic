import gymnasium as gym
from stable_baselines3 import PPO
from dual_stage_agent import DualStageAgent
from sb3_env import Env
from parameter import *
import numpy as np
from time import sleep

class DualStageEnvWrapper(gym.Env):
    def __init__(self, episode_index=0, plot=True, random_waypoint=True, agent=None, render_mode='human'):
        super(DualStageEnvWrapper, self).__init__()
        self.agent = agent
        self.render_mode = render_mode
        self.env = Env(episode_index, plot, random_waypoint, render_mode)
        self.env.set_agent(self.agent)
        self.env.agent.update_planning_state(self.env.belief_info, self.env.robot_location)
        low = -np.inf * np.ones(4, dtype=np.float32)
        high = np.inf * np.ones(4, dtype=np.float32)
        map_pixels = int(UPDATING_MAP_SIZE / CELL_SIZE)
        self.observation_space = gym.spaces.Dict({
            "belief": gym.spaces.Box(0.0, 1.0, shape=(map_pixels, map_pixels, 3), dtype=np.float32),
            "robot_state": gym.spaces.Box(low, high, shape=(4,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Box(low=np.array([0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

    def _process_belief_map(self, belief_map):
        return np.stack((
            belief_map == ROBOT_BELIEF_FREE,
            belief_map == ROBOT_BELIEF_OCCUPIED,
            belief_map == ROBOT_BELIEF_UNKNOWN
        ), axis=-1)

    def _process_obs(self, robot_state, belief_map):
        return {
            "belief": self._process_belief_map(belief_map).astype(np.float32),
            "robot_state": np.array(robot_state, dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        robot_state, _ = self.env.reset()
        return self._process_obs(robot_state, self.env.agent.updating_map_info.map), {}

    def step(self, action):
        robot_state, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_obs(robot_state, self.env.agent.updating_map_info.map)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            self.env.render()

def evaluate_model():
    # 加载训练好的模型
    # model = PPO.load("ppo_checkpoints/ppo_model_50000_steps.zip")
    model = PPO.load("ppo_sb3_final_model_4_17_can_nav.zip")  

    # 创建 agent 和环境（开启 human 渲染）
    agent = DualStageAgent(LOAD_LOCAL_CONTROLLER=False)
    env = DualStageEnvWrapper(episode_index=0, plot=False, random_waypoint=True, agent=agent, render_mode='human')

    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward
        done = terminated or truncated
        step += 1
        sleep(0.05)  # 适当延时使得渲染更流畅

    print(f"评估完成，总步数：{step}，总回报：{total_reward}")

if __name__ == "__main__":
    evaluate_model()
