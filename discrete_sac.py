from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.sac import SAC

class DiscreteSACPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.action_dim = action_space.n
        self.lr_schedule = lr_schedule

        self.actor_net = self.make_actor()
        self.actor_optimizer = self.optimizer_class(
            self.actor_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    def make_actor(self) -> nn.Module:
        actor_net = nn.Sequential()
        last_layer_dim = self.features_dim

        for idx, size in enumerate(self.net_arch):
            actor_net.add_module(f"fc{idx}", nn.Linear(last_layer_dim, size))
            actor_net.add_module(f"activation{idx}", self.activation_fn())
            last_layer_dim = size

        actor_net.add_module("action_logits", nn.Linear(last_layer_dim, self.action_dim))
        return actor_net

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs)
        logits = self.actor_net(features)
        
        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
        
        return action

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        logits = self.actor_net(features)
        distribution = torch.distributions.Categorical(logits=logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy

    def get_distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        features = self.extract_features(obs)
        logits = self.actor_net(features)
        return torch.distributions.Categorical(logits=logits)

class DiscreteCritic(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.action_dim = action_space.n
        
        self.q1_net = self._build_mlp(features_dim, net_arch, activation_fn)
        self.q2_net = self._build_mlp(features_dim, net_arch, activation_fn)

    def _build_mlp(self, input_dim: int, net_arch: List[int], activation_fn: Type[nn.Module]) -> nn.Module:
        mlp = nn.Sequential()
        last_layer_dim = input_dim

        for idx, size in enumerate(net_arch):
            mlp.add_module(f"fc{idx}", nn.Linear(last_layer_dim, size))
            mlp.add_module(f"activation{idx}", activation_fn())
            last_layer_dim = size

        mlp.add_module("q_values", nn.Linear(last_layer_dim, self.action_dim))
        return mlp

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1_values = self.q1_net(obs)
        q2_values = self.q2_net(obs)
        return q1_values, q2_values

class DiscreteSAC(SAC):
    def __init__(
        self,
        policy: Union[str, Type[DiscreteSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise = None,
        replay_buffer_class = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("DiscreteSAC only supports Discrete action spaces!")

        if policy_kwargs is None:
            policy_kwargs = dict(net_arch=[256, 256])

        # 临时修改支持的动作空间
        original_supported_spaces = BaseAlgorithm.supported_action_spaces
        BaseAlgorithm.supported_action_spaces = (spaces.Discrete,)

        try:
            super().__init__(
                policy=policy,
                env=env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                action_noise=action_noise,
                replay_buffer_class=replay_buffer_class,
                replay_buffer_kwargs=replay_buffer_kwargs,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_log,
                verbose=verbose,
                device=device,
                seed=seed,
                optimize_memory_usage=optimize_memory_usage,
                ent_coef=ent_coef,
                target_update_interval=target_update_interval,
                target_entropy=target_entropy,
            )
        finally:
            # 恢复原始的支持动作空间
            BaseAlgorithm.supported_action_spaces = original_supported_spaces

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        
        # 设置目标熵
        if isinstance(self.target_entropy, str) and self.target_entropy.startswith("auto"):
            self.target_entropy = -np.log(1.0 / self.env.action_space.n) * 0.98
        
        # 初始化熵系数
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = torch.ones(1, device=self.device) * self.ent_coef

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        for _ in range(gradient_steps):
            # 从回放缓冲区采样
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # 获取当前策略的动作分布
            features = self.policy.extract_features(replay_data.observations)
            distribution = self.policy.get_distribution(replay_data.observations)
            log_prob = distribution.log_prob(replay_data.actions)
            
            # 计算熵系数损失
            if self.ent_coef_optimizer is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            else:
                ent_coef = self.ent_coef_tensor
            
            # 计算Q值
            with torch.no_grad():
                next_features = self.policy.extract_features(replay_data.next_observations)
                next_distribution = self.policy.get_distribution(replay_data.next_observations)
                next_actions = next_distribution.sample()
                next_log_prob = next_distribution.log_prob(next_actions)
                
                next_q1, next_q2 = self.critic_target(next_features)
                next_q = torch.min(next_q1, next_q2)
                next_q = torch.gather(next_q, 1, next_actions.unsqueeze(1)).squeeze(1)
                
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * (next_q - ent_coef * next_log_prob)
            
            # 更新Critic
            current_q1, current_q2 = self.critic(features)
            current_q1 = torch.gather(current_q1, 1, replay_data.actions.unsqueeze(1)).squeeze(1)
            current_q2 = torch.gather(current_q2, 1, replay_data.actions.unsqueeze(1)).squeeze(1)
            
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            # 更新Actor
            q1, q2 = self.critic(features)
            min_q = torch.min(q1, q2)
            actor_loss = (ent_coef * log_prob - min_q.gather(1, replay_data.actions.unsqueeze(1)).squeeze(1)).mean()
            
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            
            # 更新目标网络
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            
            self._n_updates += 1
            
            # 记录日志
            self.logger.record("train/critic_loss", critic_loss.item())
            self.logger.record("train/actor_loss", actor_loss.item())
            if self.ent_coef_optimizer is not None:
                self.logger.record("train/ent_coef_loss", ent_coef_loss.item())
            self.logger.record("train/ent_coef", ent_coef.item())

    def learn(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 4,
        eval_env = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DiscreteSAC",
        eval_log_path = None,
        reset_num_timesteps: bool = True,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        observation = torch.as_tensor(observation).to(self.device)
        with torch.no_grad():
            actions = self.policy(observation, deterministic=deterministic)
        return actions.cpu().numpy(), state