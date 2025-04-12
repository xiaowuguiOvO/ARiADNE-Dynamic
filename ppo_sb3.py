import gymnasium as gym  # SB3 v2.4.1使用gymnasium而不是旧的gym
import numpy as np
from local_planner_env import LocalPlannerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os
import stable_baselines3
print(f"SB3版本: {stable_baselines3.__version__}")

class LocalPlannerEnvWrapper(gym.Env):
    def __init__(self, map_size=30.0, target_radius=0.3, max_steps=200, render_mode=None, non_blocking_render=False):
        super(LocalPlannerEnvWrapper, self).__init__()
        self.env = LocalPlannerEnv(
            map_size=map_size,
            target_radius=target_radius,
            max_steps=max_steps,
            render_mode=render_mode,
            non_blocking_render=non_blocking_render
        )
        # 根据你的环境定义正确的观察空间，这里假设 robot_state 长度为4，target_position 长度为2
        low = -np.inf * np.ones(4, dtype=np.float32)
        high = np.inf * np.ones(4, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        orig_action_space = self.env.action_space
        # 创建新的gymnasium动作空间
        self.action_space = gym.spaces.Box(
            low=np.array(orig_action_space.low, dtype=np.float32),
            high=np.array(orig_action_space.high, dtype=np.float32),
            shape=orig_action_space.shape,
            dtype=np.float32
        )
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        # 原始环境返回: next_obs, reward, terminated, truncated, info, done
        next_obs, reward, terminated, truncated, info, done = self.env.step(action)
        done = terminated or truncated
        return self._process_obs(next_obs), reward, terminated, truncated, info

    def _process_obs(self, obs):
        obs = [obs[0], obs[1], obs[2], obs[3]]
        obs = np.array(obs, dtype=np.float32)
        return obs

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

def train_with_sb3():
    # 构造包装后的环境，训练环境可以不开启渲染，评估环境开启渲染有助于观察训练效果
    env = LocalPlannerEnvWrapper(map_size=30.0, target_radius=0.3, max_steps=1000, render_mode=None)
    eval_env = LocalPlannerEnvWrapper(map_size=30.0, target_radius=0.3, max_steps=1000, render_mode='human', non_blocking_render=True)

    # 创建模型（这里使用默认的 MlpPolicy，你也可以自定义网络架构）
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_sb3_tensorboard",
                learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2)

    # 回调函数保存检查点和定期评估
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./ppo_checkpoints/',
                                             name_prefix='ppo_model')
    eval_callback = EvalCallback(eval_env, best_model_save_path='./ppo_best_model/',
                                 log_path='./ppo_eval_logs/', eval_freq=5000,
                                 deterministic=True, render=False)

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
