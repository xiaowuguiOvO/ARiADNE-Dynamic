import torch
import numpy as np
import argparse
from local_planner_env import LocalPlannerEnv
from dual_stage_model import LocalController, ValueNetwork  # 使用已有的网络结构
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
def evaluate(agent, env, num_episodes=2):
    """评估函数"""
    eval_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            input_state = [state[0], state[1], state[2], state[3]]
            # 评估时使用确定性动作
            with torch.no_grad():
                action, _ = agent.get_action(input_state)
            next_state, reward, terminated, truncated, info, done = env.step(action)
            if truncated:
                done = True
            total_reward += reward
            state = next_state
            
        eval_rewards.append(total_reward)
    
    if hasattr(env, 'close'):
        env.close()
    
    return np.mean(eval_rewards), np.std(eval_rewards)

# 设置TensorBoard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"./logs/TD3_{current_time}"
writer = SummaryWriter(log_dir)
class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device='cpu', actor_critic_class=None):
        super(PPO, self).__init__()
        if actor_critic_class is None:
            raise ValueError("Must provide actor_critic_class")

        self.device = device
        self.actor = LocalController().to(self.device)
        self.critic = ValueNetwork(state_dim).to(self.device)# Outputs state value

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.lmbda = 0.95  # for GAE
        self.entropy_coef = 0.01

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_mean = self.actor(state)
        action_std = torch.ones_like(action_mean).to(self.device) * 0.1  # fixed std
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action = action.clamp(-self.max_action, self.max_action)
        return action.cpu().detach().numpy(), dist.log_prob(action).sum(dim=-1).item()

    def evaluate_actions(self, states, actions):
        mean = self.actor(states)
        std = torch.ones_like(mean) * 0.1
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        values = self.critic(states)
        return log_probs, entropy, values

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lmbda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, trajectory, batch_size=64, epochs=10):
        states = torch.FloatTensor(np.vstack(trajectory['states'])).to(self.device)
        actions = torch.FloatTensor(np.vstack(trajectory['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(trajectory['log_probs']).unsqueeze(1).to(self.device)
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        with torch.no_grad():
            values = self.critic(states).squeeze().cpu().numpy().tolist()
            advantages = self.compute_gae(rewards, values, dones)
            returns = np.array(advantages) + np.array(values)
            advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
            returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)

        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, advantages, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for batch in dataloader:
                b_states, b_actions, b_old_log_probs, b_advantages, b_returns = batch
                log_probs, entropy, values = self.evaluate_actions(b_states, b_actions)

                ratio = torch.exp(log_probs - b_old_log_probs)
                surrogate1 = ratio * b_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                critic_loss = F.mse_loss(values, b_returns)
                entropy_loss = -self.entropy_coef * entropy.mean()

                loss = actor_loss + 0.5 * critic_loss + entropy_loss

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                self.optimizer_actor.step()
                self.optimizer_critic.step()

        
def main(args):
    # 日志记录
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"./logs/PPO_{current_time}"
    writer = SummaryWriter(log_dir)
    train_env = LocalPlannerEnv(
            map_size=5.0,
            target_radius=0.3,
            max_steps=200,
            render_mode=None  # 训练时不渲染
        )
    # 创建评估环境（有渲染）
    eval_env = LocalPlannerEnv(
        map_size=5.0,
        target_radius=0.3,
        max_steps=200,
        render_mode='human',
        non_blocking_render=True
    )
    state_dim = 4
    action_dim = 2
    max_action = float(train_env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 PPO Agent
    agent = PPO(state_dim, action_dim, max_action, device=device, actor_critic_class=LocalController)

    episode_rewards = []
    global_step = 0

    for episode in range(args.max_episodes):
        state, _ = train_env.reset()
        done = False
        total_reward = 0
        trajectory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': []
        }

        while not done:
            # print("state",state)
            input_state = [state[0], state[1], state[2], state[3]]
            action, log_prob = agent.get_action(input_state)
            # print("action",action)
            next_state, reward, terminated, truncated, info, done = train_env.step(action) #state, reward, terminated, truncated, info, done
            if truncated:
                done = True
                reward = -100
            trajectory['states'].append(input_state)
            trajectory['actions'].append(action)
            trajectory['log_probs'].append(log_prob)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)

            total_reward += reward
            state = next_state
            global_step += 1

        # 更新PPO策略
        agent.update(trajectory)
        
        # 记录训练奖励
        episode_rewards.append(total_reward)
        writer.add_scalar('Training/Episode Reward', total_reward, episode)
        
        # 每1000个episode进行评估
        # print("episode",episode)
        if episode % 200 == 0 and episode != 0:
            eval_mean, eval_std = evaluate(agent, eval_env)
            writer.add_scalar('Evaluation/Mean Reward', eval_mean, episode)
            writer.add_scalar('Evaluation/Reward Std', eval_std, episode)
            print(f"Episode {episode} | Train Reward: {total_reward:.2f} | "
                  f"Eval Mean: {eval_mean:.2f} ± {eval_std:.2f}")
        else:
            print(f"Episode {episode} | Train Reward: {total_reward:.2f}")
            
        # 保存模型检查点
        if (episode + 1) % args.save_interval == 0:
            save_path = os.path.join(log_dir, f"checkpoint_ep{episode+1}")
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'episode': episode,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
            }, os.path.join(save_path, 'model.pth'))
    
    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episodes", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=1000)
    args = parser.parse_args()
    main(args)