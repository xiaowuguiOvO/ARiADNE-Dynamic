import torch
import torch.nn as nn
import torch.nn.functional as F
from parameter import *




class WaypointSelector(nn.Module):
    """高级规划：选择下一个导航点"""
    def __init__(self, node_input_dim, embedding_dim, hidden_dim=128):
        super(WaypointSelector, self).__init__()
        
        # 节点特征编码
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 图注意力层
        self.gat1 = GraphAttention(embedding_dim, embedding_dim)
        self.gat2 = GraphAttention(embedding_dim, embedding_dim)
        
        # 输出层 - 为每个可能的节点生成分数
        self.edge_encoder = nn.Linear(1, embedding_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask):
        batch_size = node_inputs.size(0)
        
        # 编码节点特征
        node_embeddings = self.node_encoder(node_inputs)
        
        # 图注意力处理
        adj_matrix = edge_mask
        node_embeddings = self.gat1(node_embeddings, adj_matrix, node_padding_mask)
        node_embeddings = F.relu(node_embeddings)
        node_embeddings = self.gat2(node_embeddings, adj_matrix, node_padding_mask)
        
        # 获取当前节点特征
        current_node_indices = current_index.view(batch_size, 1, 1).expand(-1, -1, node_embeddings.size(2))
        current_node_features = torch.gather(node_embeddings, 1, current_node_indices).squeeze(1)
        
        # 处理边特征
        edge_features = self.edge_encoder(current_edge)
        
        # 计算候选节点分数
        k_size = current_edge.size(1)
        expanded_current_features = current_node_features.unsqueeze(1).expand(-1, k_size, -1)
        combined_features = torch.cat([expanded_current_features, edge_features], dim=2)
        
        # 生成分数并应用掩码
        logits = self.output_layer(combined_features).squeeze(-1)
        logits = logits.masked_fill(edge_padding_mask.squeeze(1) == 0, -9e15)
        
        # 转换为概率
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        return log_probs

class LocalController(nn.Module):
    """低级控制：根据局部地图和目标waypoint生成速度命令"""
    def __init__(self, node_input_dim, embedding_dim, hidden_dim=128):
        super(LocalController, self).__init__()
        
        # 节点特征编码
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 目标点编码
        self.target_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 图注意力层
        self.gat = GraphAttention(embedding_dim, embedding_dim)
        
        # 当前位置编码
        self.position_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # ----- 新增：局部地图处理 -----
        # 假设局部地图大小为 LOCAL_MAP_SIZE x LOCAL_MAP_SIZE
        LOCAL_MAP_SIZE = 32  # 局部地图大小 (如16x16或32x32像素)
        self.local_map_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 自适应池化到固定大小
        )
        
        # 扁平化后的map特征维度
        map_feature_size = 32 * 4 * 4
        
        self.map_fc = nn.Sequential(
            nn.Linear(map_feature_size, embedding_dim),
            nn.ReLU()
        )
        # ----------------------------
        
        # SAC策略网络输出层 - 注意增加了map特征的输入
        self.mean_layer = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),  # 现在是4个特征拼接
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # [vx, vy]
        )
        
        self.log_std_layer = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),  # 现在是4个特征拼接
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # [log_std_vx, log_std_vy]
        )
        
        # 动作范围限制
        self.action_scale = torch.tensor([MAX_VELOCITY, MAX_VELOCITY])
        self.action_bias = torch.tensor([0.0, 0.0])
        
    def forward(self, observation, target_waypoint, current_position, local_map, deterministic=False):
        node_inputs, node_padding_mask, edge_mask, current_index, _, _ = observation
        batch_size = node_inputs.size(0)
        
        # 编码节点特征
        node_embeddings = self.node_encoder(node_inputs)
        
        # 图注意力处理
        node_embeddings = self.gat(node_embeddings, edge_mask, node_padding_mask)
        
        # 编码目标点
        target_embedding = self.target_encoder(target_waypoint)
        
        # 编码当前位置
        position_embedding = self.position_encoder(current_position)
        
        # 获取当前节点特征
        current_node_indices = current_index.view(batch_size, 1, 1).expand(-1, -1, node_embeddings.size(2))
        current_node_features = torch.gather(node_embeddings, 1, current_node_indices).squeeze(1)
        
        # ----- 处理局部地图 -----
        # 确保local_map的形状正确 [batch_size, 1, height, width]
        if len(local_map.shape) == 3:
            local_map = local_map.unsqueeze(1)
        
        # 使用CNN处理局部地图
        map_features = self.local_map_conv(local_map)
        map_features = map_features.view(batch_size, -1)  # 扁平化
        map_embedding = self.map_fc(map_features)
        # -------------------------
        
        # 组合特征 - 现在包括地图特征
        combined_features = torch.cat([
            current_node_features, 
            target_embedding, 
            position_embedding,
            map_embedding  # 添加地图特征
        ], dim=1)
        
        # 计算动作分布参数
        mean = self.mean_layer(combined_features)
        log_std = self.log_std_layer(combined_features)
        
        # 限制log_std范围，防止方差过大或过小
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        if deterministic:
            # 确定性策略直接返回均值
            actions = mean
        else:
            # 随机策略从分布中采样
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # 重参数化技巧
            actions = torch.tanh(x_t)  # 使用tanh压缩到[-1,1]范围
            
            # 计算log_prob，用于训练
            log_prob = normal.log_prob(x_t)
            # 因为使用了tanh压缩，需要调整log_prob
            log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
        
        # 缩放到实际动作范围
        scaled_actions = actions * self.action_scale + self.action_bias
        
        if deterministic:
            return scaled_actions
        else:
            return scaled_actions, log_prob, mean, log_std

class QNetwork(nn.Module):
    """Q网络：评估状态-动作价值"""
    def __init__(self, node_input_dim, embedding_dim, action_dim=2, hidden_dim=128):
        super(QNetwork, self).__init__()
        
        # 节点特征编码
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 图注意力层
        self.gat = GraphAttention(embedding_dim, embedding_dim)
        
        # ----- 新增：局部地图处理 -----
        LOCAL_MAP_SIZE = 32  # 与控制器保持一致
        self.local_map_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 自适应池化到固定大小
        )
        
        map_feature_size = 32 * 4 * 4
        
        self.map_fc = nn.Sequential(
            nn.Linear(map_feature_size, embedding_dim),
            nn.ReLU()
        )
        # ----------------------------
        
        # 当前位置编码
        self.position_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 目标点编码
        self.target_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 动作编码
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Q值输出层 - 更新以包含所有特征
        self.q_layer = nn.Sequential(
            nn.Linear(embedding_dim * 5, hidden_dim),  # 5个特征拼接
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index, 
               current_position, target_waypoint, local_map, action):
        batch_size = node_inputs.size(0)
        
        # 编码节点特征
        node_embeddings = self.node_encoder(node_inputs)
        
        # 图注意力处理
        node_embeddings = self.gat(node_embeddings, edge_mask, node_padding_mask)
        
        # 获取当前节点特征
        current_node_indices = current_index.view(batch_size, 1, 1).expand(-1, -1, node_embeddings.size(2))
        current_node_features = torch.gather(node_embeddings, 1, current_node_indices).squeeze(1)
        
        # 编码位置
        position_embedding = self.position_encoder(current_position)
        
        # 编码目标点
        target_embedding = self.target_encoder(target_waypoint)
        
        # 处理局部地图
        if len(local_map.shape) == 3:
            local_map = local_map.unsqueeze(1)
        map_features = self.local_map_conv(local_map)
        map_features = map_features.view(batch_size, -1)
        map_embedding = self.map_fc(map_features)
        
        # 编码动作
        action_embedding = self.action_encoder(action)
        
        # 组合特征
        combined_features = torch.cat([
            current_node_features,
            position_embedding,
            target_embedding,
            map_embedding,
            action_embedding
        ], dim=1)
        
        # 计算Q值
        q_value = self.q_layer(combined_features)
        
        return q_value