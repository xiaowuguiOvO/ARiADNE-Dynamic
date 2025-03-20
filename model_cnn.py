import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNPolicy(nn.Module):
    def __init__(self, map_size, hidden_dim=64):
        super(CNNPolicy, self).__init__()
        
        # 输入通道：2（vx,vy）+ 1（occupancy）= 3
        self.cnn_encoder = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            
            # 第二层卷积
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            
            # 第三层卷积
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
        )
        
        # 计算输出特征图大小
        feature_size = map_size * map_size * hidden_dim*4
        
        # 策略头（输出动作概率）
        self.policy_head = nn.Sequential(
            nn.Linear(feature_size, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, 2)  # 输出vx, vy
        )
        
        # 值函数头（如果需要）
        self.value_head = nn.Sequential(
            nn.Linear(feature_size, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, 1)
        )
        
    def forward(self, pedestrian_map, occupancy_map):
        # pedestrian_map: [batch_size, 2, map_size, map_size]
        # occupancy_map: [batch_size, 1, map_size, map_size]
        
        # 合并输入
        x = torch.cat([pedestrian_map, occupancy_map], dim=1)
        
        # CNN编码
        features = self.cnn_encoder(x)
        features_flat = features.view(features.size(0), -1)
        
        # 获取动作概率和值函数
        action_logits = self.policy_head(features_flat)
        value = self.value_head(features_flat)
        
        return action_logits, value