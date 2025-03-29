import torch
import torch.nn as nn
<<<<<<< HEAD
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
=======
import math
from parameter import *

# a pointer network layer for policy output
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            U = U.masked_fill(mask == 1, -1e8)
        attention = torch.log_softmax(U, dim=-1)  # n_batch*n_query*n_key

        return attention


# standard multi head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(k_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(v_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U)  # copy for n_heads times

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, -1e8)

        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        # out = heads.permute(1, 2, 0, 3).reshape(n_batch, n_query, n_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(-1, n_query, self.embedding_dim)

        return out, attention  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)
        h, _ = self.multiHeadAttention(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multiHeadAttention(q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask,
                                       attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, w


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w
>>>>>>> Continues
