import gymnasium as gym  # SB3 v2.4.1使用gymnasium而不是旧的gym
import numpy as np
from local_planner_env import LocalPlannerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os
import stable_baselines3
from sb3_env import Env
from dual_stage_agent import DualStageAgent
from stable_baselines3.common.vec_env import DummyVecEnv

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
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=np.array([0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        # 重置环境 随机选一张地图
        obs, _ = self.env.reset()
        return obs, {}

    def step(self, action):
        # 用 action 控制机器人移动
        # print("action",action)
        # print("obs", self.env.agent.get_robot_state())
        action = [1, 0.3]
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self):
        # print("render")
        if self.render_mode == 'human':
            self.env.render()  # 你自己内部的渲染逻辑
        pass

    def close(self):
        pass

def train_with_sb3():
    # 构造包装后的环境，训练环境可以不开启渲染，评估环境开启渲染有助于观察训练效果
    agent = DualStageAgent(LOAD_LOCAL_CONTROLLER=False)
    env = DualStageEnvWrapper(episode_index=0, plot=False, random_waypoint=True, agent=agent, render_mode=None)
    eval_env = DummyVecEnv([lambda: DualStageEnvWrapper(episode_index=0, plot=False, random_waypoint=True, agent=agent, render_mode='human')])

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_sb3_tensorboard",
                learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,ent_coef=0.01)

    # 回调函数保存检查点和定期评估
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./ppo_checkpoints/',
                                             name_prefix='ppo_model')

    eval_callback = EvalCallback(eval_env, best_model_save_path='./ppo_best_model/',
                                 log_path='./ppo_eval_logs/', eval_freq=100,
                                 deterministic=True, render=True, n_eval_episodes=1)
    
    # 训练
    total_timesteps = 100000  # 根据需要调整训练步数
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    
    # 保存最终模型
    model.save("ppo_sb3_final_model")
    print("训练结束，模型已保存。")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    train_with_sb3()
