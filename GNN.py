import math
import numpy as np
import random
from collections import deque
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import pickle
import datetime

# 导入 PyTorch Geometric 相关模块
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# 1) 环境定义：网络部署环境
###############################################################################
class NetworkDeploymentEnv(gym.Env):
    def __init__(self):
        super(NetworkDeploymentEnv, self).__init__()
        self.coverage_map = None
        self.connections_map = None
        self.load_precomputed_data()

        # 地图与网格尺寸
        self.map_size = 1000
        self.grid_size = 1000
        self.node_spacing = 50
        self.n_potential_nodes_per_row = int(self.grid_size // self.node_spacing)
        self.n_potential_nodes = (self.grid_size // self.node_spacing) ** 2

        # 动作与观测空间
        self.action_space = spaces.Discrete(self.n_potential_nodes + 1)  # +1 代表 "do nothing"
        self.coverage_grid = np.zeros((self.map_size, self.map_size), dtype=np.int8)
        self.observation_space = spaces.Dict({
            "D": spaces.MultiBinary(self.n_potential_nodes),
            "R": spaces.Box(low=0, high=50, shape=(self.n_potential_nodes,)),
            "N": spaces.Box(low=0, high=30, shape=(self.n_potential_nodes,))
        })

        # 网络参数
        self.overhead = 1.2
        self.node_data_rate = 2
        self.donor_data_rate = 50
        self.coverage_radius = 200
        self.backhaul_radius = 300
        self.narrow = 1
        self.numberofdonor = 5
        self.min_inbound_links = 2  # 至少两条入链

        # 运行时状态
        self.current_step = 0
        self.max_steps = 60
        self.previous_actions = set()
        self.last_reward = 0
        self.coverage_needs_update = True

        # 邻接矩阵：仍按 "每个 donor 一张" 保存（最小改动方案）
        self.donor_adjacency_matrices = np.zeros((self.numberofdonor,
                                                  self.n_potential_nodes,
                                                  self.n_potential_nodes), dtype=np.int8)
        self.donor_indices = self.get_fixed_donor_positions()
        self.coverage_threshold = 98
        self.reset()

    # ---------------------------------------------------------------------
    # 数据加载与基础工具
    # ---------------------------------------------------------------------
    def load_precomputed_data(self):
        """加载预计算的覆盖点和邻居表"""
        with open('coverage_map.pkl1002', 'rb') as f:
            self.coverage_map = pickle.load(f)
        with open('connections_map.pkl1002', 'rb') as f:
            self.connections_map = pickle.load(f)
        print("[Env] coverage_map & connections_map loaded.")

    def get_fixed_donor_positions(self):
        """五点骰布局"""
        p = self.n_potential_nodes_per_row
        return [
            (p // 2) * p + (p // 2),                          # center
            (p // 4) * p + (p // 4),                          # TL
            (p // 4) * p + (3 * p // 4),                      # TR
            (3 * p // 4) * p + (p // 4),                      # BL
            (3 * p // 4) * p + (3 * p // 4)                   # BR
        ]

    def find_nearest_donor(self, node_idx):
        """返回几何距离最近的 donor id"""
        node_pos = self.get_node_position(node_idx)
        distances = [
            (d_id, self.get_distance(node_pos, self.get_node_position(donor_idx)))
            for d_id, donor_idx in enumerate(self.donor_indices)
        ]
        nearest_id = min(distances, key=lambda x: x[1])[0]
        return nearest_id

    # ---------------------------------------------------------------------
    # Gym 核心接口
    # ---------------------------------------------------------------------
    def reset(self):
        self.state = {
            "D": np.zeros(self.n_potential_nodes, dtype=np.int8),
            "R": np.zeros(self.n_potential_nodes, dtype=np.float32),
            "N": np.zeros(self.n_potential_nodes, dtype=np.float32)
        }
        self.coverage_grid.fill(0)
        self.donor_adjacency_matrices.fill(0)

        # 初始化 5 个 donor
        for idx in self.donor_indices:
            self.state["D"][idx] = 1
            self.state["R"][idx] = self.donor_data_rate
            self.update_coverage_single_node(idx)

        self.current_step = 0
        self.previous_actions.clear()
        self.last_reward = 0
        self.coverage_needs_update = True
        self.previous_coverage = 0.0

        print(f"[Env Reset] coverage={self.calculate_coverage_percentage():.2f}% (start)")
        return self._get_flattened_state()

    def step(self, action_index):
        coverage_before = self.calculate_coverage_percentage()
        valid_actions = self.get_valid_actions()
        if action_index not in valid_actions:
            action_index = random.choice(valid_actions)
            print(f"[Env Step] Invalid action chosen randomly: {action_index}")

        reward = 0.0
        done = False
        node_index = action_index

        if node_index in self.previous_actions and node_index != self.n_potential_nodes:
            reward = -1.0  # 重复动作惩罚
        else:
            if node_index < self.n_potential_nodes:
                reward += self.deploy_node(node_index)
            else:
                reward += self.keep_node()

        coverage_after = self.calculate_coverage_percentage()
        print(f"[Env Step] Step={self.current_step}, Action={action_index}, Reward={reward:.2f}, Coverage={coverage_after:.2f}%")

        self.previous_actions.add(node_index)
        self.last_reward = reward
        self.current_step += 1

        # 终止条件
        if self.current_step >= self.max_steps or coverage_after >= self.coverage_threshold:
            done = True
            self.previous_actions.clear()

        return self._get_flattened_state(), reward, done, {}

    # ---------------------------------------------------------------------
    # 节点部署相关
    # ---------------------------------------------------------------------
    def deploy_node(self, node_index):
        """尝试部署节点，成功则更新覆盖和链路"""
        if self.state["D"][node_index] == 1:
            return self.calculate_reward()  # 已部署

        connected, donor_id, inbound_links = self.reconnect_node(node_index, self.min_inbound_links)
        if not connected:
            print(f"[Env Deploy] Node {node_index} failed: {len(inbound_links)} inbound < {self.min_inbound_links}")
            return self.calculate_reward() 

        # 正式部署
        self.state["D"][node_index] = 1
        self.update_coverage_single_node(node_index)
        for target in inbound_links:
            self.donor_adjacency_matrices[donor_id, target, node_index] = 1
            self.state["N"][target] += 1
        self.update_data_rates(donor_id)
        return self.calculate_reward() 

    def keep_node(self):
        return self.calculate_reward()

    def reconnect_node(self, node_index, min_links):
        """允许跨 donor：从所有已部署节点中选速率最高的若干条入链"""
        min_data_rate = 2.4
        candidates = [
            (target, self.state['R'][target])
            for target in self.connections_map.get(node_index, [])
            if self.state['D'][target] == 1 and self.state['R'][target] >= min_data_rate
        ]
        if not candidates:
            return False, None, []
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in candidates[:min_links]]
        if len(selected) < min_links:
            return False, None, selected
        donor_id = self.find_nearest_donor(selected[0])
        # 继承最高速率
        self.state['R'][node_index] = self.state['R'][selected[0]]
        print(f"[Env Reconnect] Node {node_index} connected to {len(selected)} targets (cross-donor allowed)")
        return True, donor_id, selected

    # ---------------------------------------------------------------------
    # 覆盖、数据速率、奖励
    # ---------------------------------------------------------------------
    def update_coverage_single_node(self, node_index):
        if self.state["D"][node_index] == 1 and node_index in self.coverage_map:
            for (x, y) in self.coverage_map[node_index]:
                self.coverage_grid[x, y] = 1

    def update_data_rates(self, donor_id):
        """仍沿用 "每 donor 一张邻接矩阵" 的简易带宽分享"""
        adj = self.donor_adjacency_matrices[donor_id]
        donor_idx = self.donor_indices[donor_id]
        connected_nodes = set()
        for node in range(self.n_potential_nodes):
            if adj[node, :].sum() > 0 or adj[:, node].sum() > 0 or node == donor_idx:
                connected_nodes.add(node)
        num_conn = len(connected_nodes)
        shared_rate = max(0, self.donor_data_rate - (num_conn * self.overhead * self.node_data_rate))
        for n in connected_nodes:
            self.state['R'][n] = shared_rate
        return True

    # ---------------------------------------------------------------------
    # 新版 reward：阈值化覆盖、翻倍节点惩罚、超额覆盖罚分、韧性罚分、终局奖惩
    # 调用方式：rew = self.calculate_reward(done)  ← 在 step() 内把 done 传进来
    # ---------------------------------------------------------------------
    def calculate_reward(self, done: bool = False) -> float:
        # ---- 基本量 ------------------------------------------------------
        current_cov   = self.calculate_coverage_percentage() / 100.0         # [0,1]
        theta_cov     = self.coverage_threshold / 100.0                      # 0.95 / 0.98 …
        deployed      = max(0, self.total_deployed_nodes() - len(self.donor_indices))

        # ---- 参数（可视需要外提为 self.*） --------------------------------
        C_big         = 20.0     # 达标固定奖励
        C_miss        = 40.0     # 覆盖缺口惩罚系数
        lambda_N      = 1.5      # 每部署一节点惩罚
        P_miss        = 4.0      # 入链不足惩罚
        alpha_over    = 10.0     # 超额覆盖惩罚系数
        E_bonus       = 50.0     # 终局成功奖励
        E_fail        = 50.0     # 终局失败惩罚

        # ---- 覆盖奖励 / 惩罚 --------------------------------------------
        if current_cov >= theta_cov:
            coverage_reward   = C_big
            over_cover_penalty = alpha_over * (current_cov - theta_cov)
        else:
            coverage_reward   = -C_miss * (theta_cov - current_cov)
            over_cover_penalty = 0.0

        # ---- 节点惩罚 ----------------------------------------------------
        deploy_penalty = lambda_N * deployed

        # ---- 韧性惩罚：入链不足 m 条 -------------------------------------
        nodes_with_few_links = 0
        for node_idx in range(self.n_potential_nodes):
            if self.state["D"][node_idx] and node_idx not in self.donor_indices:
                inbound = sum(self.donor_adjacency_matrices[d_id, :, node_idx].sum()
                              for d_id in range(self.numberofdonor))
                if inbound < self.min_inbound_links:
                    nodes_with_few_links += 1
        resilience_penalty = P_miss * nodes_with_few_links

        # ---- 汇总即时 reward --------------------------------------------
        reward = (coverage_reward
                  - deploy_penalty
                  - resilience_penalty
                  - over_cover_penalty)

        # ---- 终局奖 / 罚 -------------------------------------------------
        if done:
            reward += (E_bonus if current_cov >= theta_cov else -E_fail)

        return reward

    # ---------------------------------------------------------------------
    # 工具函数
    # ---------------------------------------------------------------------
    def update_coverage(self):
        for n, deployed in enumerate(self.state['D']):
            if deployed and n in self.coverage_map:
                for (x, y) in self.coverage_map[n]:
                    self.coverage_grid[x, y] = 1

    def get_valid_actions(self):
        valid_nodes = set(range(self.n_potential_nodes)) - set(np.where(self.state['D'] == 1)[0])
        resilient_nodes = []
        for node in valid_nodes:
            valid_targets = [t for t in self.connections_map.get(node, [])
                             if self.state['D'][t] == 1 and self.state['R'][t] >= 2.4]
            if len(valid_targets) >= self.min_inbound_links:
                resilient_nodes.append(node)
        return resilient_nodes + [self.n_potential_nodes]

    def total_deployed_nodes(self):
        return int(self.state['D'].sum())

    def calculate_coverage_percentage(self):
        return self.coverage_grid.sum() / (self.grid_size * self.grid_size) * 100

    def get_distance(self, pos1, pos2):
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def get_node_position(self, node_index):
        row = node_index // self.n_potential_nodes_per_row
        col = node_index % self.n_potential_nodes_per_row
        x = int(col * self.node_spacing * self.narrow)
        y = int(row * self.node_spacing * self.narrow)
        return (x, y)

    def _get_flattened_state(self):
        return np.concatenate([self.state['D'], self.state['R'], self.state['N']])

    # ---------------------------------------------------------------------
    # 图构造（修正距离过滤条件 <= backhaul_radius）
    # ---------------------------------------------------------------------
    def construct_graph(self, max_neighbors_per_node=5, min_data_rate_threshold=2.0):
        num_nodes = self.n_potential_nodes
        deployment_status = self.state['D'].reshape(-1, 1)
        data_rate_norm = (self.state['R'] / self.donor_data_rate).reshape(-1, 1)
        inbound_links = np.zeros((num_nodes, 1))
        for node in range(num_nodes):
            if node not in self.donor_indices:
                inbound = sum(self.donor_adjacency_matrices[d, :, node].sum() for d in range(self.numberofdonor))
                inbound_links[node, 0] = min(inbound / 2.0, 1.0)
        node_features = np.concatenate([deployment_status, data_rate_norm, inbound_links], axis=1)
        x = torch.tensor(node_features, dtype=torch.float32, device=device)

        edge_pairs = []
        for node in range(num_nodes):
            neighbors = self.connections_map.get(node, [])
            neighbors = [t for t in neighbors if not (self.state['D'][node] == 1 and self.state['D'][t] == 1)]
            neighbors.sort(key=lambda t: self.state['R'][t], reverse=True)
            selected = []
            for t in neighbors:
                if len(selected) >= max_neighbors_per_node:
                    break
                if self.state['R'][t] >= min_data_rate_threshold:
                    pos1, pos2 = self.get_node_position(node), self.get_node_position(t)
                    if self.get_distance(pos1, pos2) <= self.backhaul_radius:
                        selected.append(t)
            for t in selected:
                edge_pairs.append([node, t])
        edge_index = torch.tensor(edge_pairs, dtype=torch.long, device=device).t().contiguous()
        node_type = torch.zeros(num_nodes, dtype=torch.long, device=device)
        for idx in self.donor_indices:
            node_type[idx] = 1
        data = Data(x=x, edge_index=edge_index)
        data.node_type = node_type
        return data

###############################################################################
# 2) GNN 网络设计
###############################################################################
class GNNPolicy(nn.Module):
    def __init__(self, node_feature_dim=3, hidden_dim=32):
        super(GNNPolicy, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc_candidate1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc_candidate2 = nn.Linear(hidden_dim//2, 1)
        self.fc_do_nothing1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc_do_nothing2 = nn.Linear(hidden_dim//2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, node_type = data.x, data.edge_index, data.node_type
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        candidate_mask = (node_type == 0)
        candidate_embeddings = x[candidate_mask]
        candidate_hidden = F.relu(self.fc_candidate1(candidate_embeddings))
        candidate_scores = self.fc_candidate2(candidate_hidden).squeeze(-1)
        importance_weights = F.softmax(torch.sum(x, dim=1), dim=0).unsqueeze(1)
        global_embedding = torch.sum(x * importance_weights, dim=0, keepdim=True)
        do_nothing_hidden = F.relu(self.fc_do_nothing1(global_embedding))
        do_nothing_score = self.fc_do_nothing2(do_nothing_hidden).squeeze()
        logits = torch.cat([candidate_scores, do_nothing_score.unsqueeze(0)], dim=0)
        candidate_indices = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
        return logits, candidate_indices

class GNNCritic(nn.Module):
    def __init__(self, node_feature_dim=3, hidden_dim=32):
        super(GNNCritic, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc_candidate1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc_candidate2 = nn.Linear(hidden_dim//2, 1)
        self.fc_do_nothing1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc_do_nothing2 = nn.Linear(hidden_dim//2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, node_type = data.x, data.edge_index, data.node_type
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        candidate_mask = (node_type == 0)
        candidate_embeddings = x[candidate_mask]
        candidate_hidden = F.relu(self.fc_candidate1(candidate_embeddings))
        candidate_q = self.fc_candidate2(candidate_hidden).squeeze(-1)
        importance_weights = F.softmax(torch.sum(x, dim=1), dim=0).unsqueeze(1)
        global_embedding = torch.sum(x * importance_weights, dim=0, keepdim=True)
        do_nothing_hidden = F.relu(self.fc_do_nothing1(global_embedding))
        do_nothing_q = self.fc_do_nothing2(do_nothing_hidden).squeeze()
        q_values = torch.cat([candidate_q, do_nothing_q.unsqueeze(0)], dim=0)
        candidate_indices = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
        return q_values, candidate_indices

###############################################################################
# 3) GNN 离散 SAC Agent
###############################################################################
class GNNDiscreteSACAgent:
    def __init__(self, n_potential_nodes, learning_rate=3e-5, gamma=0.99, alpha=0.2,
                 buffer_size=100000, batch_size=32):
        self.n_potential_nodes = n_potential_nodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=buffer_size)

        self.policy = GNNPolicy().to(device)
        self.q1 = GNNCritic().to(device)
        self.q2 = GNNCritic().to(device)
        self.q1_target = GNNCritic().to(device)
        self.q2_target = GNNCritic().to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        self.tau = 0.01
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate*2)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate*2)
        self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = -np.log(1.0 / (n_potential_nodes + 1))
        self.alpha = self.log_alpha.exp()

        self.total_steps = 0
        self.policy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.policy_optimizer, mode='min', factor=0.5, patience=200, verbose=True)
        self.q1_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.q1_optimizer, mode='min', factor=0.5, patience=200, verbose=True)
        self.q2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.q2_optimizer, mode='min', factor=0.5, patience=200, verbose=True)

    def select_action(self, graph, valid_actions):
        self.policy.eval()
        with torch.no_grad():
            logits, candidate_indices = self.policy(graph)
        self.policy.train()
        candidate_logits = []
        candidate_global_indices = []
        for i, global_idx in enumerate(candidate_indices.cpu().numpy()):
            if global_idx in valid_actions:
                candidate_logits.append(logits[i])
                candidate_global_indices.append(global_idx)
        do_nothing_logit = logits[-1].unsqueeze(0)
        if candidate_logits:
            final_logits = torch.cat([torch.stack(candidate_logits), do_nothing_logit], dim=0)
        else:
            final_logits = do_nothing_logit
        final_probs = F.softmax(final_logits, dim=0)
        dist = torch.distributions.Categorical(final_probs)
        action_idx = dist.sample().item()
        if candidate_logits and action_idx < len(candidate_logits):
            action = candidate_global_indices[action_idx]
        else:
            action = self.n_potential_nodes
        return action

    def store_transition(self, state_array, action, reward, next_state_array, done):
        self.replay_buffer.append((state_array, action, reward, next_state_array, done))

    def reconstruct_graph(self, state_array, env):
        env.state["D"], env.state["R"], env.state["N"] = np.split(state_array, 3)
        return env.construct_graph()

    def soft_update(self, online_net, target_net):
        for param, target_param in zip(online_net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update_alpha(self, log_pi):
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        return alpha_loss.item()

    def experience_replay(self, env):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)
        state_arrays, actions, rewards, next_state_arrays, dones = zip(*batch)

        graphs = [self.reconstruct_graph(state_array, env) for state_array in state_arrays]
        next_graphs = [self.reconstruct_graph(next_state_array, env) for next_state_array in next_state_arrays]

        rewards = torch.tensor(rewards, dtype=torch.float, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)

        with torch.no_grad():
            target_q_values = []
            for next_graph in next_graphs:
                q1_next, _ = self.q1_target(next_graph)
                q2_next, _ = self.q2_target(next_graph)
                logits_next, _ = self.policy(next_graph)
                probs_next = F.softmax(logits_next, dim=0)
                log_probs_next = F.log_softmax(logits_next, dim=0)
                q_min_next = torch.min(q1_next, q2_next)
                soft_value = (probs_next * (q_min_next - self.alpha * log_probs_next)).sum().unsqueeze(0)
                target_q_values.append(soft_value)
            target_q = rewards + (1 - dones) * self.gamma * torch.stack(target_q_values)

        total_q1_loss = 0.0
        total_q2_loss = 0.0
        for graph, action, target in zip(graphs, actions, target_q):
            q1_values, candidate_indices = self.q1(graph)
            q2_values, _ = self.q2(graph)
            candidate_logits = []
            candidate_global_indices = []
            for i, global_idx in enumerate(candidate_indices.cpu().numpy()):
                candidate_logits.append(q1_values[i])
                candidate_global_indices.append(global_idx)
            do_nothing_q1 = q1_values[-1].unsqueeze(0)
            if candidate_logits:
                final_q1 = torch.cat([torch.stack(candidate_logits), do_nothing_q1], dim=0)
            else:
                final_q1 = do_nothing_q1
            if action == self.n_potential_nodes:
                q1_val = final_q1[-1]
            else:
                if action in candidate_global_indices:
                    idx = candidate_global_indices.index(action)
                    q1_val = final_q1[idx]
                else:
                    q1_val = final_q1[-1]

            candidate_logits2 = []
            candidate_global_indices2 = []
            for i, global_idx in enumerate(candidate_indices.cpu().numpy()):
                candidate_logits2.append(q2_values[i])
                candidate_global_indices2.append(global_idx)
            do_nothing_q2 = q2_values[-1].unsqueeze(0)
            if candidate_logits2:
                final_q2 = torch.cat([torch.stack(candidate_logits2), do_nothing_q2], dim=0)
            else:
                final_q2 = do_nothing_q2
            if action == self.n_potential_nodes:
                q2_val = final_q2[-1]
            else:
                if action in candidate_global_indices2:
                    idx = candidate_global_indices2.index(action)
                    q2_val = final_q2[idx]
                else:
                    q2_val = final_q2[-1]

            total_q1_loss += F.mse_loss(q1_val, target.squeeze())
            total_q2_loss += F.mse_loss(q2_val, target.squeeze())

        total_q1_loss /= self.batch_size
        total_q2_loss /= self.batch_size

        self.q1_optimizer.zero_grad()
        total_q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        total_q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
        self.q2_optimizer.step()

        policy_loss_total = 0.0
        log_pi_total = 0.0
        for graph in graphs:
            logits, _ = self.policy(graph)
            probs = F.softmax(logits, dim=0)
            log_probs = F.log_softmax(logits, dim=0)
            q1_values, _ = self.q1(graph)
            q2_values, _ = self.q2(graph)
            q_min = torch.min(q1_values, q2_values)
            policy_loss = (probs * (self.alpha * log_probs - q_min)).sum()
            policy_loss_total += policy_loss
            log_pi_total += (probs * log_probs).sum()
        policy_loss_total /= self.batch_size
        log_pi_total /= self.batch_size

        self.policy_optimizer.zero_grad()
        policy_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        alpha_loss = self.update_alpha(log_pi_total)

        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)
        if self.total_steps % 1000 == 0:
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
            print(f"[Agent] Hard target update at step {self.total_steps}")

        self.policy_scheduler.step(policy_loss_total.item())
        self.q1_scheduler.step(total_q1_loss.item())
        self.q2_scheduler.step(total_q2_loss.item())

        avg_loss = 0.5 * (total_q1_loss.item() + total_q2_loss.item()) + policy_loss_total.item() + alpha_loss
        return avg_loss

###############################################################################
# 4) 训练函数和主函数
###############################################################################
def train(env, agent, episodes=6000):
    episode_rewards = []
    episode_losses = []
    coverage_per_episode = []
    deployed_nodes_per_episode = []
    all_steps_details = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        done = False
        sum_loss = 0.0
        loss_steps = 0

        graph = env.construct_graph()

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(graph, valid_actions)
            next_state, reward, done, _ = env.step(action)
            next_graph = env.construct_graph()

            step_details = {
                'Episode': ep,
                'Step': step_count,
                'Action': action,
                'Reward': reward,
                'Coverage': env.calculate_coverage_percentage()
            }
            all_steps_details.append(step_details)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.experience_replay(env)
            if loss != 0.0:
                sum_loss += loss
                loss_steps += 1
            graph = next_graph
            state = next_state
            total_reward += reward
            step_count += 1
            agent.total_steps += 1

        coverage_at_end = env.calculate_coverage_percentage()
        coverage_per_episode.append(coverage_at_end)
        episode_rewards.append(total_reward)
        avg_loss = sum_loss / (loss_steps if loss_steps > 0 else 1)
        episode_losses.append(avg_loss)
        deployed_nodes = env.total_deployed_nodes()
        deployed_nodes_per_episode.append(deployed_nodes)

        print(f"[Train] Episode={ep}, Steps={step_count}, Reward={total_reward:.2f}, Coverage={coverage_at_end:.2f}%, Loss={avg_loss:.4f}")
        torch.cuda.empty_cache()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    step_csv_name = f'step_details_gnn_{timestamp}.csv'
    with open(step_csv_name, mode='w', newline='') as file:
        fieldnames = ["Episode", "Step", "Action", "Reward", "Coverage"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for detail in all_steps_details:
            writer.writerow(detail)
    episode_csv_name = f'episode_summary_gnn_{timestamp}.csv'
    with open(episode_csv_name, mode='w', newline='') as file:
        fieldnames = ["Episode", "Coverage", "DeployedNodes"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for ep in range(len(coverage_per_episode)):
            writer.writerow({
                "Episode": ep,
                "Coverage": coverage_per_episode[ep],
                "DeployedNodes": deployed_nodes_per_episode[ep]
            })

    return episode_rewards, episode_losses, coverage_per_episode, deployed_nodes_per_episode

def plot_moving_window(data, window_size=300):
    smoothed = []
    cum_sum = 0.0
    queue = deque()
    for val in data:
        queue.append(val)
        cum_sum += val
        if len(queue) > window_size:
            cum_sum -= queue.popleft()
        smoothed.append(cum_sum / len(queue))
    return smoothed

def main():
    donor_shape = "5-dice"
    env = NetworkDeploymentEnv()
    example_state = env.reset()
    state_dim = len(example_state)
    action_dim = env.action_space.n

    agent = GNNDiscreteSACAgent(n_potential_nodes=env.n_potential_nodes, learning_rate=1e-5,
                                gamma=0.99, alpha=0.2, buffer_size=50000, batch_size=32)

    pretrained_model_path = f"gnn_{donor_shape}_transfer.pth"
    if os.path.exists(pretrained_model_path):
        print("[Main] Pre-trained model found, loading...")
        used_transfer_learning = True
    else:
        print("[Main] No pre-trained model found, training from scratch.")
        used_transfer_learning = False

    episodes = 6000
    rewards, losses, coverages, deployed_nodes = train(env, agent, episodes=episodes)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(rewards)), rewards, label='Reward per Episode', alpha=0.4)
    smoothed_rewards = plot_moving_window(rewards, window_size=300)
    plt.plot(range(len(rewards)), smoothed_rewards, label='Moving Avg (300)', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode (GNN)')
    plt.legend()
    plt.savefig(f'Reward_vs_Episode_gnn_{donor_shape}_{"transfer" if used_transfer_learning else "from_scratch"}.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(losses)), losses, label='Loss per Episode', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss vs Episode (GNN)')
    plt.legend()
    plt.savefig(f'Loss_vs_Episode_gnn_{donor_shape}_{"transfer" if used_transfer_learning else "from_scratch"}.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(coverages)), coverages, label='Coverage per Episode', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Coverage (%)')
    plt.title('Coverage vs Episode (GNN)')
    plt.legend()
    plt.savefig(f'Coverage_vs_Episode_gnn_{donor_shape}_{"transfer" if used_transfer_learning else "from_scratch"}.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(deployed_nodes)), deployed_nodes, label='Deployed Nodes per Episode', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Deployed Nodes')
    plt.title('Deployed Nodes vs Episode (GNN)')
    plt.legend()
    plt.savefig(f'DeployedNodes_vs_Episode_gnn_{donor_shape}_{"transfer" if used_transfer_learning else "from_scratch"}.png')
    plt.close()

    save_filename = f"gnn_{donor_shape}_{'transfer' if used_transfer_learning else 'from_scratch'}.pth"
    print(f"[Main] Training complete. Model saved to {save_filename}")

if __name__ == "__main__":
    main()