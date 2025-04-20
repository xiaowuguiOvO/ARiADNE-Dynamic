import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道维度计算平均值和最大值
        avg_out = th.mean(x, dim=1, keepdim=True)
        max_out, _ = th.max(x, dim=1, keepdim=True)
        # 拼接特征
        x_cat = th.cat([avg_out, max_out], dim=1)
        # 应用卷积和sigmoid激活
        attention_map = self.sigmoid(self.conv(x_cat))
        # 应用注意力
        return x * attention_map

class BeliefFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super(BeliefFeatureExtractor, self).__init__(observation_space, features_dim)

        # Belief 是图像 → 使用更深的CNN结构
        self.belief_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),  # 85x85 → 41x41
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # → 20x20
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # → 9x9
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # → 9x9
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # 添加空间注意力机制
        self.spatial_attention = SpatialAttention(kernel_size=5)
        
        # 最后进行平均池化和展平，大幅减少特征维度
        self.global_pool = nn.AdaptiveAvgPool2d((3, 3))
        
        # 计算CNN输出后的特征维度
        belief_output_dim = 64 * 3 * 3  # 576，比原来的6400小得多
        
        # 为机器人状态单独添加一个小型MLP
        self.robot_state_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        
        # 组合特征的多层感知机
        self.combined_mlp = nn.Sequential(
            nn.Linear(belief_output_dim + 32, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim)
        )

    def forward(self, obs):
        # 处理信念地图
        belief = obs["belief"].permute(0, 3, 1, 2)  # NHWC → NCHW
        belief_feat = self.belief_cnn(belief)
        belief_feat = self.spatial_attention(belief_feat)
        belief_feat = self.global_pool(belief_feat)
        belief_feat = belief_feat.flatten(1)
        
        # 处理机器人状态
        robot_state = obs["robot_state"]
        robot_feat = self.robot_state_mlp(robot_state)
        
        # 组合特征
        combined = th.cat((belief_feat, robot_feat), dim=1)
        return self.combined_mlp(combined)
