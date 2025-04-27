import torch
import torch.nn as nn
from dual_stage_model import Encoder, Decoder

class WaypointSelectorSAC(nn.Module):
    def __init__(self, node_dim, embedding_dim, action_dim):
        super(WaypointSelectorSAC, self).__init__()
        # 基础编码器部分保持不变
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.viewpoint_encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        self.current_node_decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # SAC特有的输出层：均值和标准差
        self.mean_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        self.log_std_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        self.log_std_min = -20
        self.log_std_max = 2

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_viewpoint_feature = self.viewpoint_encoder(
            src=node_feature, 
            key_padding_mask=node_padding_mask, 
            attn_mask=edge_mask
        )
        return enhanced_viewpoint_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(
            enhanced_node_feature, 1,
            current_index.repeat(1, 1, embedding_dim)
        )
        enhanced_current_node_feature, _ = self.current_node_decoder(
            current_node_feature,
            enhanced_node_feature,
            node_padding_mask
        )
        return current_node_feature, enhanced_current_node_feature

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask):
        # 编码图结构
        enhanced_viewpoint_feature = self.encode_graph(
            node_inputs, node_padding_mask, edge_mask)
        
        # 解码当前状态
        current_node_feature, enhanced_current_node_feature = self.decode_state(
            enhanced_viewpoint_feature, current_index, node_padding_mask)
        
        # 组合特征
        combined_feature = self.current_embedding(
            torch.cat((enhanced_current_node_feature, current_node_feature), dim=-1))
        
        # 输出动作分布参数
        mean = self.mean_head(combined_feature)
        log_std = self.log_std_head(combined_feature)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

class WaypointCritic(nn.Module):
    def __init__(self, node_dim, embedding_dim, action_dim):
        super(WaypointCritic, self).__init__()
        # 状态编码器
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.viewpoint_encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        self.current_node_decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # 动作编码器
        self.action_encoder = nn.Linear(action_dim, embedding_dim)
        
        # Q值输出层
        self.q_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_viewpoint_feature = self.viewpoint_encoder(
            src=node_feature,
            key_padding_mask=node_padding_mask,
            attn_mask=edge_mask
        )
        return enhanced_viewpoint_feature

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index, actions):
        # 编码状态
        enhanced_viewpoint_feature = self.encode_graph(
            node_inputs, node_padding_mask, edge_mask)
        
        # 编码动作
        action_embedding = self.action_encoder(actions)
        
        # 组合特征
        combined_feature = torch.cat([enhanced_viewpoint_feature, action_embedding], dim=-1)
        
        # 输出Q值
        q_value = self.q_head(combined_feature)
        return q_value