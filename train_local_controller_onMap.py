from dual_stage_model import LocalController, ControllerQNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
# from local_planner_env import LocalPlannerEnv
from dual_stage_env import Env
import time
import numpy as np
import random
from collections import deque
from dual_stage_agent import DualStageAgent
import argparse
import datetime

# 设置TensorBoard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"./logs/TD3_{current_time}"
writer = SummaryWriter(log_dir)

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, device='cpu', agent=None):
        if agent is None:
            raise ValueError("Agent must be provided")
            
        self.device = device
        # 确保 actor 在正确的设备上
        self.actor = agent.local_controller.to(device)
        
        # 确保所有网络都在同一个设备上
        self.actor_target = LocalController(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = ControllerQNetwork(state_dim, action_dim).to(device)
        self.critic_target = ControllerQNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -float('inf')
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(self.device)
            next_state = torch.Tensor(batch_next_states).to(self.device)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)
            done = torch.Tensor(batch_dones).to(self.device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        
# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
eval_freq = 5e3  # After how many steps to perform the evaluation
max_ep = 200  # maximum number of steps per episode
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 5e6  # Maximum number of steps to perform
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = (
    1000  # Number of steps over which the initial exploration noise will decay over
)
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 40  # Size of the mini-batch
discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # Frequency of Actor network updates
buffer_size = 1e6  # Maximum size of the buffer
file_name = "TD3_velodyne"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
load_model = False  # Weather to load a stored model
load_model_path = None  # 要加载的模型路径，设为None则不加载
start_episode = 0  # 如果加载模型，从哪个episode开始计数
random_near_obstacle = True  # To take random actions near obstacles or not

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--load-model', type=str, help='要加载的模型路径')
parser.add_argument('--start-episode', type=int, default=0, help='起始episode编号')
parser.add_argument('--no-render', action='store_true', help='禁用渲染，用于加速训练')
args = parser.parse_args()

# 如果指定了命令行参数，优先使用命令行参数
if args.load_model:
    load_model = True
    load_model_path = args.load_model
    start_episode = args.start_episode
    print(f"将从模型 {load_model_path} 继续训练，起始episode为 {start_episode}")

# 根据命令行参数决定是否渲染
render = not args.no_render

# Create the network storage folders
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Create the training environment
MAX_TIME_STEPS = 5e6
environment_dim = 20
robot_dim = 4
render_mode = 'human' if render else None
env = Env(plot=True)
# time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = robot_dim
action_dim = 2
max_action = 1

# Create the network
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)

timestep = 0
episode_timesteps = 0
episode_num = start_episode  # 从指定的episode开始
total_reward = 0
# init
agent = DualStageAgent(device=device)
network = TD3(state_dim, action_dim, max_action, device=device,agent=agent)

# 如果需要加载模型
if load_model and load_model_path:
    try:
        network.load(f"episode_{start_episode}", load_model_path)
        print(f"成功加载模型：{load_model_path}/episode_{start_episode}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        # 如果是best模型
        try:
            network.load("best", load_model_path)
            print(f"成功加载最佳模型：{load_model_path}/best")
        except Exception as e:
            print(f"加载最佳模型也失败: {e}")
            print("将使用新初始化的模型开始训练")

obs, info = env.reset()
print(f"init obs: {obs}")
x, y, theta, v_linear, v_angular, distance_to_target = obs['robot_state']
current_state = np.array([distance_to_target, theta, v_linear, v_angular], dtype=np.float32)
current_state = torch.tensor(current_state, dtype=torch.float32)
print(f"current_state: {current_state}")

done = True
# 定义保存模型的间隔(每多少个episode保存一次)
save_episode_interval = 10
# 记录训练信息
training_info = {
    "episode_rewards": [],
    "episode_steps": [],
    "best_reward": -float('inf'),
    "best_episode": 0
}
# 确保保存模型的目录存在
if not os.path.exists("./saved_models"):
    os.makedirs("./saved_models")
    
while timestep < MAX_TIME_STEPS:
    
    if done:
        if timestep != 0:
            print(f"Episode {episode_num}: 总步数={episode_timesteps}, 总奖励={total_reward:.2f}")
            # 记录本次episode的信息
            training_info["episode_rewards"].append(total_reward)
            training_info["episode_steps"].append(episode_timesteps)
            
            # 记录到TensorBoard
            writer.add_scalar("Train/Episode_Reward", total_reward, episode_num)
            writer.add_scalar("Train/Episode_Steps", episode_timesteps, episode_num)
            writer.add_scalar("Train/Exploration_Noise", expl_noise, episode_num)
            
            # 如果有足够的数据，计算平均奖励
            if len(training_info["episode_rewards"]) >= 10:
                avg_reward = sum(training_info["episode_rewards"][-10:]) / 10
                writer.add_scalar("Train/Avg_Reward_10", avg_reward, episode_num)
            
            # 检查是否是最佳表现
            if total_reward > training_info["best_reward"]:
                training_info["best_reward"] = total_reward
                training_info["best_episode"] = episode_num
                # 保存最佳模型
                best_model_path = f"./saved_models/best_model"
                if not os.path.exists(best_model_path):
                    os.makedirs(best_model_path)
                network.save("best", best_model_path)
                print(f"新的最佳模型已保存! Episode {episode_num}, 奖励 {total_reward:.2f}")
                
                # 记录最佳表现
                writer.add_scalar("Train/Best_Reward", total_reward, episode_num)
            
            print(f"训练网络中...")
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )
            
            # 每隔固定episode保存一次模型

        env.reset()
        done = False
        episode_timesteps = 0
        episode_num += 1
        total_reward = 0
        
    current_state = agent.get_robot_state()
    current_state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        action = agent.local_controller(current_state)
        action = action.cpu().numpy().squeeze()
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((expl_noise - expl_min) / expl_decay_steps)
        
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action)
    
    # action = np.array([1, 0.1])
    # print(f"action: {action}")
    obs, reward, terminated, truncated, info, done = env.step(action)
    
    # print(f"reward: {reward}")
    
    # 检查是否到达了最大episode步数
    # if episode_timesteps + 1 >= max_ep and not done:
    #     reward -= 100.0
    #     print(f"Episode {episode_num} reached max steps ({max_ep}) without reaching target. Adding penalty.")
    
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    x, y, theta, v_linear, v_angular, distance_to_target = obs['robot_state']
    agent.update_robot_state(distance_to_target, theta, v_linear, v_angular)
    next_state = np.array([distance_to_target, theta, v_linear, v_angular], dtype=np.float32)
    
    current_state = current_state.cpu().numpy().squeeze()
    replay_buffer.add(current_state, action, reward, done_bool, next_state)
    # print(current_state, next_state)
    # print(f"len of replay buffer: {replay_buffer.size()}")
    timestep += 1
    episode_timesteps += 1
    total_reward += reward
    if terminated or truncated:
        break

# 训练结束，保存最终模型
final_model_path = f"./saved_models/final_model"
if not os.path.exists(final_model_path):
    os.makedirs(final_model_path)
network.save("final", final_model_path)
print(f"训练结束，最终模型已保存")

# 保存完整的训练记录
final_training_stats = {
    "total_episodes": episode_num,
    "total_timesteps": timestep,
    "best_reward": training_info["best_reward"],
    "best_episode": training_info["best_episode"],
    "episode_rewards": training_info["episode_rewards"],
    "episode_steps": training_info["episode_steps"]
}

import json
with open(f"./saved_models/training_history.json", 'w') as f:
    json.dump(final_training_stats, f, indent=4)

# 关闭TensorBoard
writer.close()

print(f"训练完成! 共完成 {episode_num} 个episodes, {timestep} 个时间步")
print(f"最佳奖励: {training_info['best_reward']:.2f}, 在第 {training_info['best_episode']} 个episode")