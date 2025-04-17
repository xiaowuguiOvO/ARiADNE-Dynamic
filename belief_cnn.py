import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym

class BeliefFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super(BeliefFeatureExtractor, self).__init__(observation_space, features_dim)

        # Belief 是图像 → CNN
        self.belief_cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2),  # 85x85 → 41x41
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # → 20x20
            nn.ReLU(),
            nn.Flatten()
        )

        belief_output_dim = 16 * 20 * 20  # 自己根据输出shape计算

        # Robot State 是 4D 向量 → 直接 MLP
        self.linear = nn.Sequential(
            nn.Linear(belief_output_dim + 4, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        belief = obs["belief"].permute(0, 3, 1, 2)  # NHWC → NCHW
        belief_feat = self.belief_cnn(belief)
        robot_state = obs["robot_state"]
        combined = th.cat((belief_feat, robot_state), dim=1)
        return self.linear(combined)
