import torch
import torch.nn as nn
import torch.nn.functional as F
from parameter import *
import math
    
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

class WaypointSelector(nn.Module):
    def __init__(self, node_dim, embedding_dim):
        super(WaypointSelector, self).__init__()
        # viewpoint encoder, encode viewpoints graph into viewpoint feature
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.viewpoint_encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        # current node decoder, decode current node feature from enhanced viewpoint feature
        self.current_node_decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # pointer network, select waypoint from current node feature
        self.pointer_network = SingleHeadAttention(embedding_dim)
        
    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_viewpoint_feature = self.viewpoint_encoder(src=node_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)
        return enhanced_viewpoint_feature
    
    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1,
                                                    current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.current_node_decoder(current_node_feature,
                                                                    enhanced_node_feature,
                                                                    node_padding_mask)

        return current_node_feature, enhanced_current_node_feature
    
    def output_logp(self, current_node_feature, enhanced_current_node_feature,
                      enhanced_node_feature, current_edge, edge_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        # current_state_feature = current_node_feature
        current_state_feature = self.current_embedding(torch.cat((enhanced_current_node_feature,
                                                                current_node_feature), dim=-1))

        neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                           current_edge.repeat(1, 1, embedding_dim))

        logp = self.pointer_network(current_state_feature, neighboring_feature, edge_padding_mask)
        logp = logp.squeeze(1)

        return logp
    
    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask):
        # encode graph
        enhanced_viewpoint_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        current_node_feature, enhanced_current_node_feature = self.decode_state(enhanced_viewpoint_feature, current_index, node_padding_mask)
        # select waypoint
        waypoint_logp = self.output_logp(current_node_feature, enhanced_current_node_feature,
                                          enhanced_viewpoint_feature, current_edge, edge_padding_mask)
        return waypoint_logp
    

class LocalController(nn.Module):
    def __init__(self, state_dim=8, hidden_dim=128):
        """
        底层运动控制器，负责生成到达目标路点的速度指令
        
        参数:
            state_dim: 状态维度(机器人状态+目标路点信息)
            hidden_dim: 隐藏层维度
        """
        super(LocalController, self).__init__()
        
        # 特征提取网络
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值网络（确定性部分）
        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2)  # 输出线速度和角速度
        )
        
        # 标准差网络（随机部分，用于探索）
        self.log_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2)
        )
        
    def prepare_state(self, robot_state, target_waypoint):
        """
        准备控制器输入状态
        
        参数:
            robot_state: [x, y, theta, v_linear, v_angular, ...]
            target_waypoint: [x, y]
        """
        # 计算机器人到目标的向量
        robot_pos = robot_state[:, :2]  # 机器人位置[x,y]
        pos_diff = target_waypoint - robot_pos  # 位置差向量
        
        # 计算距离和方向
        dist = torch.norm(pos_diff, dim=1, keepdim=True)  # 欧氏距离
        target_angle = torch.atan2(pos_diff[:, 1], pos_diff[:, 0]).unsqueeze(1)  # 目标角度
        
        # 计算与当前朝向的角度差
        heading = robot_state[:, 2].unsqueeze(1)  # 当前朝向角
        angle_diff = target_angle - heading
        # 归一化角度差到[-π, π]范围
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        # 组合完整状态
        full_state = torch.cat([
            dist,                  # 到目标的距离
            angle_diff,            # 朝向与目标方向的角度差
            robot_state[:, 3:5],   # 当前线速度和角速度
            robot_state,           # 完整机器人状态
            target_waypoint        # 目标路点
        ], dim=1)
        
        return full_state
        
    def forward(self, robot_state, target_waypoint):
        """
        前向传播，生成动作
        
        返回:
            velocity: 线速度和角速度命令
            mean: 动作均值
            log_std: 动作对数标准差
        """
        # 准备状态输入
        state = self.prepare_state(robot_state, target_waypoint)
        
        # 提取特征
        features = self.feature_net(state)
        
        # 计算动作均值和标准差
        mean = self.mean_net(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        std = torch.exp(log_std)
        
        # 在训练时采样动作
        if self.training:
            # 从正态分布采样
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # 重参数化技巧
            # 使用tanh压缩范围，提高训练稳定性
            y_t = torch.tanh(x_t)
            # 将动作映射到实际的速度范围
            linear_vel = (y_t[:, 0] + 1) / 2 * MAX_LINEAR_VELOCITY  # [0, MAX]
            angular_vel = y_t[:, 1] * MAX_ANGULAR_VELOCITY  # [-MAX, MAX]
            velocity = torch.stack([linear_vel, angular_vel], dim=1)
            return velocity, mean, log_std
        else:
            # 推理时直接使用均值
            linear_vel = (torch.tanh(mean[:, 0]) + 1) / 2 * MAX_LINEAR_VELOCITY
            angular_vel = torch.tanh(mean[:, 1]) * MAX_ANGULAR_VELOCITY
            velocity = torch.stack([linear_vel, angular_vel], dim=1)
            return velocity, mean, log_std
    
    def get_action_and_logprob(self, robot_state, target_waypoint):
        """
        计算动作和对应的对数概率
        """
        state = self.prepare_state(robot_state, target_waypoint)
        features = self.feature_net(state)
        
        mean = self.mean_net(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        
        # 计算对数概率
        log_prob = normal.log_prob(x_t)
        # 由于tanh变换，需要调整对数概率
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # 映射到实际速度
        linear_vel = (y_t[:, 0] + 1) / 2 * MAX_LINEAR_VELOCITY
        angular_vel = y_t[:, 1] * MAX_ANGULAR_VELOCITY
        velocity = torch.stack([linear_vel, angular_vel], dim=1)
        
        return velocity, log_prob, mean, log_std
    

class WayPointQNet(nn.Module):
    def __init__(self, node_dim, embedding_dim):
        super(WayPointQNet, self).__init__()

        # local graph encoder
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)

        # decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        self.q_values_layer = nn.Linear(embedding_dim * 2, 1)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature,
                                                         key_padding_mask=node_padding_mask,
                                                         attn_mask=edge_mask)

        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1,
                                                  current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(current_node_feature,
                                                                    enhanced_node_feature,
                                                                    node_padding_mask)

        return current_node_feature, enhanced_current_node_feature

    def output_q(self, current_node_feature, enhanced_current_node_feature, enhanced_node_feature,
                 current_edge, edge_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        k_size = current_edge.size()[1]
        # current_state_feature = current_node_feature
        current_state_feature = self.current_embedding(torch.cat((enhanced_current_node_feature,
                                                                 current_node_feature), dim=-1))

        neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                           current_edge.repeat(1, 1, embedding_dim))

        action_features = torch.cat((current_state_feature.repeat(1, k_size, 1), neighboring_feature), dim=-1)
        q_values = self.q_values_layer(action_features)
        return q_values

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        current_node_feature, enhanced_current_node_feature = self.decode_state(enhanced_node_feature, current_index, node_padding_mask)
        q_values = self.output_q(current_node_feature, enhanced_current_node_feature,
                                 enhanced_node_feature, current_edge, edge_padding_mask)

        return q_values

class ControllerQNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=128):
        """
        底层控制器的Q网络
        评估状态-动作对的价值函数
        
        参数:
            state_dim: 状态维度(机器人状态+目标路点)
            action_dim: 动作维度(线速度和角速度)
            hidden_dim: 隐藏层维度
        """
        super(ControllerQNetwork, self).__init__()
        
        # 双Q网络结构，提高训练稳定性
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 状态预处理，与LocalController共享
        self.prepare_state = LocalController().prepare_state
        
    def forward(self, robot_state, target_waypoint, action):
        """
        计算状态-动作对的Q值
        
        参数:
            robot_state: 机器人状态 [batch_size, state_dim]
            target_waypoint: 目标路点 [batch_size, 2]
            action: 速度命令 [batch_size, 2]
            
        返回:
            q1, q2: 两个Q网络的估计值
        """
        # 准备状态表示
        state = self.prepare_state(robot_state, target_waypoint)
        
        # 连接状态和动作
        x = torch.cat([state, action], dim=1)
        
        # 计算两个Q值估计
        q1 = self.q1(x)
        q2 = self.q2(x)
        
        return q1, q2
    
    def min_q(self, robot_state, target_waypoint, action):
        """返回两个Q网络估计的最小值，用于保守估计"""
        q1, q2 = self.forward(robot_state, target_waypoint, action)
        return torch.min(q1, q2)
    

# class DualStageAgent(nn.Module):
#     """整合高层规划和底层控制的完整双阶段代理"""
    
#     def __init__(self, node_dim, robot_state_dim, embedding_dim=128):
#         super(DualStageAgent, self).__init__()
        
#         # 高层路点选择器
#         self.waypoint_selector = WaypointSelector(node_dim, embedding_dim)
        
#         # 底层运动控制器
#         self.local_controller = LocalController(state_dim=robot_state_dim+4)  # +4是距离和角度差
        
#         # 训练模式标志
#         self.training_high_level = True  # 控制训练哪一层
        
#     def forward(self, observation):
#         """
#         执行完整的双阶段决策过程
        
#         参数:
#             observation: 包含图表示和机器人状态的观测字典
            
#         返回:
#             waypoint: 选择的下一个路点
#             velocity: 生成的速度命令
#         """
#         # 1. 高层规划 - 选择路点
#         waypoint_logits = self.waypoint_selector(
#             observation['node_inputs'],
#             observation['node_padding_mask'],
#             observation['edge_mask'],
#             observation['current_index'],
#             observation['current_edge'],
#             observation['edge_padding_mask']
#         )
        
#         # 根据logits选择路点 (训练时采样，推理时取最大值)
#         if self.training:
#             waypoint_dist = torch.distributions.Categorical(logits=waypoint_logits)
#             waypoint_idx = waypoint_dist.sample()
#         else:
#             waypoint_idx = torch.argmax(waypoint_logits, dim=1)
            
#         # 提取选择的路点坐标 (假设前两维是xy坐标)
#         selected_waypoint = torch.gather(
#             observation['node_inputs'], 1,
#             waypoint_idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, 2)
#         ).squeeze(1)[:, :2]
        
#         # 2. 底层控制 - 生成速度命令
#         robot_state = observation['robot_state']
#         velocity, mean, log_std = self.local_controller(robot_state, selected_waypoint)
        
#         return {
#             'waypoint_idx': waypoint_idx,
#             'waypoint': selected_waypoint,
#             'velocity': velocity,
#             'waypoint_logits': waypoint_logits,
#             'velocity_mean': mean,
#             'velocity_log_std': log_std
#         }