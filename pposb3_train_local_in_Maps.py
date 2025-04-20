import gymnasium as gym
from gymnasium.envs.registration import register as _register
gym.register = _register
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from dual_stage_agent import DualStageAgent
from sb3_env import Env, DualStageEnvWrapper
from belief_cnn import BeliefFeatureExtractor
import argparse
from parameter import *
import os

# 3. 定义一个小工厂函数，不再调用 env.seed，而是通过 reset(seed=) 来传递随机数
def make_env(rank, seed=0):
    def _init():
        agent = DualStageAgent(LOAD_LOCAL_CONTROLLER=False)
        env = DualStageEnvWrapper(
            episode_index=rank,
            plot=False,
            random_waypoint=True,
            agent=agent,
            render_mode=None
        )
        # 通过 reset 传递 seed
        env.reset(seed=seed + rank)
        return env
    return _init

# 5. 并行训练主体
def train_with_sb3():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--total-timesteps', type=int, default=1_000_000)
    args = parser.parse_args()

    # 并行环境
    env = SubprocVecEnv([make_env(i, seed=100) for i in range(args.n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

    # 并行评估环境
    eval_basic_env = SubprocVecEnv([make_env(0, seed=200)])
    if os.path.exists("vec_normalize.pkl"):
        # 加载已有的标准化统计
        eval_env = VecNormalize.load("vec_normalize.pkl", eval_basic_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        # 文件不存在，创建一个只标准化 obs 的 eval_env
        eval_env = VecNormalize(eval_basic_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.)


    # PPO
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.96,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./ppo_tb"
    )
    cb_ckpt = CheckpointCallback(10000, "./checkpoints", "ppo", save_vecnormalize=True)
    cb_eval = EvalCallback(
    eval_env,
    best_model_save_path="./best",   # 必须用关键字参数
    log_path="./best_logs/",         # 可选
    eval_freq=5000,
    n_eval_episodes=1,
    render=False,
    verbose=1
    )

    model.learn(total_timesteps=args.total_timesteps, callback=[cb_ckpt, cb_eval])
    model.save("ppo_final")
    env.save("vec_normalize.pkl")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    train_with_sb3()