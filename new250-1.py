import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to avoid display issues
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import csv
import datetime
import time
import pickle

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)

        # Value and Advantage streams with two layers
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean())
        return q_values


class Agent:
    def __init__(self, state_dim, action_dim, learning_rate=0.0001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=60000, alpha=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DuelingDQN(state_dim, action_dim).to(device)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1000)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=80000)
        self.total_steps = 0
        self.taken_actions = set()

        self.alpha = alpha
        self.reward_mean = 0
        self.reward_std = 1

    def select_action(self, state, valid_actions):
        state = self.normalize_state(state)
        valid_actions = [action for action in valid_actions if action not in self.taken_actions]

        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1.0 * self.total_steps / self.epsilon_decay
        )

        if random.random() < eps_threshold or not valid_actions:
            # 随机探索
            action_index = random.choice(list(set(range(self.action_dim)) - self.taken_actions))
        else:
            # 贪心选择
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state).squeeze().cpu().detach().numpy()
            valid_q_values = {action: q_values[action] for action in valid_actions}
            action_index = max(valid_q_values, key=valid_q_values.get)

        self.taken_actions.add(action_index)
        return action_index

    def reset_actions(self):
        self.taken_actions.clear()

    def store_transition(self, state, action, reward, next_state, done):
        reward = self.normalize_reward(reward)
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states, dtype=np.float32), device=device, dtype=torch.float32)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = F.mse_loss(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        current_lr = self.get_current_lr()
        self.scheduler.step(loss)
        updated_lr = self.get_current_lr()

        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def normalize_state(self, state):
        state = np.array(state)
        return (state - state.min()) / (state.max() - state.min() + 1e-5)

    def normalize_reward(self, reward):
        self.reward_mean = self.alpha * reward + (1 - self.alpha) * self.reward_mean
        self.reward_std = self.alpha * (reward - self.reward_mean) ** 2 + (1 - self.alpha) * self.reward_std
        return (reward - self.reward_mean) / (np.sqrt(self.reward_std) + 1e-5)

    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1.0 * self.total_steps / self.epsilon_decay
        )

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


def train(env, agent, episodes, batch_size=512, target_update=64, scale=300, save_model_interval=30000):
    episode_rewards = []
    episode_losses = []
    data = {
        "Episode": [],
        "Total Reward": [],
        "Avg Loss": [],
        "Deployed Nodes": [],
        "Coverage": []
    }
    all_steps_details = []

    for episode in range(episodes):
        start_episode_time = time.time()
        state = env.reset()
        agent.reset_actions()
        total_reward = 0
        total_loss = 0
        done = False
        step_count = 0

        while not done:
            start_step_time = time.time()
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.experience_replay(batch_size)

            state = next_state
            total_reward += reward
            total_loss += loss if loss is not None else 0
            step_count += 1
            agent.total_steps += 1

            step_details = {
                "Episode": episode,
                "Step": step_count,
                "Action": action,
                "Reward": reward,
                "Coverage": env.calculate_coverage_percentage()
            }
            all_steps_details.append(step_details)

            if agent.total_steps % target_update == 0:
                agent.update_target_network()

            end_step_time = time.time()

            if step_count >= max_steps:
                done = True
            elif env.calculate_coverage_percentage() >= env.coverage_threshold:
                done = True

        end_episode_time = time.time()

        avg_loss = total_loss / step_count if step_count > 0 else 0
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        deployed_nodes = env.total_deployed_nodes()
        coverage_percentage = env.calculate_coverage_percentage()
        epsilon = agent.get_epsilon()
        current_lr = agent.get_current_lr()

        data["Episode"].append(episode)
        data["Total Reward"].append(total_reward)
        data["Avg Loss"].append(avg_loss)
        data["Deployed Nodes"].append(deployed_nodes)
        data["Coverage"].append(coverage_percentage)

        # 周期性保存模型
        if episode % save_model_interval == 0 and episode > 0:
            torch.save(agent.model.state_dict(), f'model_episode_{episode}.pth')
            print(f'Model saved at episode {episode}')

        print(f'End of Episode {episode}:')
        print(f'Total Reward: {total_reward}')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Total Deployed Nodes: {deployed_nodes}')
        print(f'Coverage: {coverage_percentage:.2f}')
        print(f'Epsilon: {epsilon:.4f}')
        print(f'Learning Rate: {current_lr:.6f}\n')

    # 训练结束，保存最终模型
    torch.save(agent.model.state_dict(), '250-1.pth')
    print('Final model saved.')

    # 计算滑动统计
    avg_rewards_per_scale_episodes = [
        np.mean(episode_rewards[i:i + scale]) for i in range(0, len(episode_rewards), scale)
    ]
    avg_losses_per_scale_episodes = [
        np.mean(episode_losses[i:i + scale]) for i in range(0, len(episode_losses), scale)
    ]
    avg_numofnodes_per_scale_episodes = [
        np.mean(data["Deployed Nodes"][i:i + scale]) for i in range(0, len(data["Deployed Nodes"]), scale)
    ]
    avg_coverage_per_scale_episodes = [
        np.mean(data["Coverage"][i:i + scale]) for i in range(0, len(data["Coverage"]), scale)
    ]
    episodes_scale = list(range(0, len(episode_rewards), scale))

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'all_episodes_step_details_{timestamp}.csv'
    with open(filename, mode='w', newline='') as file:
        fieldnames = ["Episode", "Step", "Action", "Reward", "Coverage"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for detail in all_steps_details:
            writer.writerow(detail)

    # Plot Reward
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Reward per Episode')
    plt.plot(episodes_scale, avg_rewards_per_scale_episodes, label='Avg Reward per 100 Episodes', color='red',
             linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Episode vs Reward')
    plt.legend()
    filename = f'Episode_vs_Reward_{timestamp}.png'
    plt.savefig(filename)
    plt.close()

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(episode_losses, label='Loss per Episode')
    plt.plot(episodes_scale, avg_losses_per_scale_episodes, label='Avg Loss per 100 Episodes', color='blue',
             linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Episode vs Loss')
    plt.legend()
    filename = f'Episode_vs_Loss_{timestamp}.png'
    plt.savefig(filename)
    plt.close()

    # Plot Deployed Nodes and Coverage
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(data["Episode"], data["Deployed Nodes"], label='Deployed Nodes per Episode', color='green', marker='o',
             linestyle='-', linewidth=1, markersize=4)
    plt.plot(episodes_scale, avg_numofnodes_per_scale_episodes, label='Avg nodes per 100 Episodes', color='red',
             linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Number of Deployed Nodes')
    plt.title('Episode vs Number of Deployed Nodes')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(data["Episode"], data["Coverage"], label='Coverage Percentage per Episode', color='purple', marker='o',
             linestyle='-', linewidth=1, markersize=4)
    plt.plot(episodes_scale, avg_coverage_per_scale_episodes, label='Avg Coverage per 100 Episodes', color='red',
             linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Coverage Percentage')
    plt.title('Episode vs Coverage Percentage')
    plt.legend()

    plt.tight_layout()
    filename = f'Episode_vs_Coverage_Percentage_{timestamp}.png'
    plt.savefig(filename)
    plt.close()

    # Save training data to CSV
    filename = f'training_data_{timestamp}.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Reward', 'Avg Loss', 'Deployed Nodes', 'Coverage'])
        for i in range(len(data['Episode'])):
            writer.writerow(
                [
                    data['Episode'][i],
                    data['Total Reward'][i],
                    data['Avg Loss'][i],
                    data['Deployed Nodes'][i],
                    data['Coverage'][i]
                ]
            )

    return episode_rewards, episode_losses


def print_initial_R_N(state, n_potential_nodes):
    R_start = n_potential_nodes
    N_start = 2 * n_potential_nodes


class NetworkDeploymentEnv(gym.Env):
    def __init__(self):
        super(NetworkDeploymentEnv, self).__init__()
        self.coverage_map = None
        self.connections_map = None
        self.load_precomputed_data()

        # 基本参数
        self.map_size = 250
        self.grid_size = 250
        self.node_spacing = 25
        self.n_potential_nodes_per_row = self.grid_size // self.node_spacing
        self.n_potential_nodes = self.n_potential_nodes_per_row ** 2
        self.action_space = self.n_potential_nodes + 1
        self.coverage_grid = np.zeros((self.map_size, self.map_size), dtype=np.int8)

        self.observation_space = spaces.Dict({
            "D": spaces.MultiBinary(self.n_potential_nodes),
            "R": spaces.Box(low=0, high=30, shape=(self.n_potential_nodes,)),
            "N": spaces.Box(low=0, high=20, shape=(self.n_potential_nodes,))
        })

        # 网络相关参数
        self.overhead = 1.2
        self.node_data_rate = 2
        self.donor_data_rate = 30
        self.coverage_radius = 50
        self.backhaul_radius = 75
        self.narrow = 1

        # Donor 及步数设定
        self.numberofdonor = numberofdonor
        self.current_step = 0
        self.max_steps = max_steps
        self.previous_actions = set()
        self.last_reward = None
        self.coverage_needs_update = True

        # Donor adjacency + 其他变量
        self.donor_adjacency_matrices = np.zeros((self.numberofdonor, self.n_potential_nodes, self.n_potential_nodes))
        self.previous_coverage = 0.0
        self.optimal_node_count = 24
        self.node_penalty_factor = 20
        self.coverage_threshold = 100

        # ---------------------------
        # 在这里获取 donor 索引并修复越界问题
        self.donor_indices = self.get_fixed_donor_positions()
        # ---------------------------

        self.reset()

    def get_fixed_donor_positions(self):
        # 计算中心行、列
        row_center = self.n_potential_nodes_per_row // 2
        col_center = self.n_potential_nodes_per_row // 2

        # 原先给出的五边形偏移
        pentagon_offsets = [
            (0, -7),  # top
            (-2, 4),  # Upper right
            (2, 4),  # Lower right
            (5, -4),  # Lower left
            (-5, -4)  # Upper left
        ]

        pentagon_positions = []
        for (dr, dc) in pentagon_offsets:
            r = row_center + dr
            c = col_center + dc
            # 只有在不越界的情况下才加入
            if 0 <= r < self.n_potential_nodes_per_row and 0 <= c < self.n_potential_nodes_per_row:
                node_index = r * self.n_potential_nodes_per_row + c
                pentagon_positions.append(node_index)

        # ----------------------
        # 如果实际加进去的点没达到 5 个，可以再人工补几个点
        # ----------------------
        required_donors = 5
        if len(pentagon_positions) < required_donors:
            # 先获取所有可能的 node_index
            all_indices = list(range(self.n_potential_nodes))
            # 去除已经加进去的
            candidates = list(set(all_indices) - set(pentagon_positions))
            # 随机打乱
            random.shuffle(candidates)
            # 依次补充直到长度达 5
            while len(pentagon_positions) < required_donors and candidates:
                pentagon_positions.append(candidates.pop())

        return pentagon_positions

    def load_precomputed_data(self):
        try:
            with open('coverage_map.pkl250', 'rb') as f:
                self.coverage_map = pickle.load(f)
            with open('connections_map.pkl250', 'rb') as f:
                self.connections_map = pickle.load(f)
        except FileNotFoundError:
            print("Failed to load the precomputed data files. Please check the files' existence and paths.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def reset(self):
        self.state = {
            "D": np.zeros(self.n_potential_nodes),
            "R": np.zeros(self.n_potential_nodes),
            "N": np.zeros(self.n_potential_nodes)
        }
        self.coverage_grid.fill(0)
        self.donor_adjacency_matrices.fill(0)

        # 部署 donor 节点
        for idx in self.donor_indices:
            self.state["D"][idx] = 1
            self.state["R"][idx] = 30
            self.update_coverage_single_node(idx)

        self.current_step = 0
        self.previous_actions = set()
        self.last_reward = 0
        self.coverage_needs_update = True
        self.no_improvement_steps = 0  # 初始化累计无增长步数

        return self._get_flattened_state()

    def update_coverage_single_node(self, node_index):
        if self.state["D"][node_index] == 1:
            for (x, y) in self.coverage_map[node_index]:
                self.coverage_grid[x, y] = 1

    def step(self, action_index):
        coverage_before = self.calculate_coverage_percentage()
        valid_actions = self.get_valid_actions()
        if action_index not in valid_actions:
            action_index = valid_actions[np.random.randint(len(valid_actions))]

        rewards = 0
        done = False
        node_index = action_index

        if node_index in self.previous_actions:
            rewards += self.last_reward
        else:
            if node_index < self.n_potential_nodes:
                rewards += self.deploy_node(node_index)
            else:
                rewards += self.keep_node()

        self.previous_actions.add(node_index)
        self.last_reward = rewards
        self.current_step += 1
        self.coverage_needs_update = False

        coverage_after = self.calculate_coverage_percentage()
        if coverage_after > coverage_before:
            self.no_improvement_steps = 0
        else:
            self.no_improvement_steps += 1
            # 连续3步无增长且覆盖不达标，施加一次 penalty
            if self.no_improvement_steps >= 3 and coverage_after < self.coverage_threshold:
                penalty = -100
                rewards += penalty
                self.no_improvement_steps = 0

        if self.current_step >= max_steps:
            self.current_step = 0
            self.coverage_needs_update = True
            done = True
        elif coverage_after >= self.coverage_threshold:
            self.current_step = 0
            self.coverage_needs_update = True
            rewards += 50
            done = True

        return self._get_flattened_state(), rewards, done

    def deploy_node(self, node_index):
        node_x, node_y = self.get_node_position(node_index)
        coverage_before = np.sum(self.coverage_grid)

        if self.state["D"][node_index] == 1:
            self.update_coverage()
            return self.calculate_reward()
        else:
            connected, donor_id = self.reconnect_node(node_index)
            if not connected:
                self.state["D"][node_index] = 0
                self.update_coverage()
                return self.calculate_reward()

            self.state["D"][node_index] = 1
            self.update_coverage_single_node(node_index)
            coverage_after = np.sum(self.coverage_grid)
            coverage_increase = coverage_after - coverage_before

            return self.calculate_reward()

    def keep_node(self):
        return self.calculate_reward()

    def reconnect_node(self, node_index):
        for d_id, donor_index in enumerate(self.donor_indices):
            best_target = None
            max_data_rate = 2.4
            for target in self.connections_map.get(node_index, []):
                if self.state['D'][target] == 1 and self.state['R'][target] > max_data_rate:
                    if target not in self.donor_indices or target == donor_index:
                        max_data_rate = self.state['R'][target]
                        best_target = target

            if best_target is not None:
                self.state["N"][best_target] += 1
                self.state["R"][best_target] -= self.node_data_rate * self.overhead
                self.state["R"][node_index] = self.state["R"][best_target]
                self.donor_adjacency_matrices[d_id, best_target, node_index] = 1
                self.update_data_rates(d_id)
                return True, d_id

        return False, None

    def update_data_rates(self, donor_id):
        adjacency_matrix = self.donor_adjacency_matrices[donor_id]
        donor_index = self.donor_indices[donor_id]

        connected_nodes = set()
        for node_index in range(self.n_potential_nodes):
            if (adjacency_matrix[donor_index, node_index] > 0 or
                adjacency_matrix[node_index, donor_index] > 0 or
                np.sum(adjacency_matrix[:, node_index]) > 0 or
                np.sum(adjacency_matrix[node_index, :]) > 0):
                connected_nodes.add(node_index)

        num_connected_nodes = len(connected_nodes)
        shared_data_rate = max(
            0,
            self.donor_data_rate - (num_connected_nodes * self.overhead * self.node_data_rate)
        )

        for node_index in connected_nodes:
            self.state["R"][node_index] = shared_data_rate
        return True

    def calculate_uncovered_areas(self):
        uncovered_positions = []
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.coverage_grid[i, j] == 0:
                    uncovered_positions.append((i, j))
        return uncovered_positions

    def get_valid_actions(self):
        valid_nodes = set(range(self.n_potential_nodes))
        deployed_nodes = set(np.where(self.state["D"] == 1)[0])
        valid_nodes = valid_nodes - deployed_nodes

        deployed_node_positions = [self.get_node_position(node_index) for node_index in deployed_nodes]
        coverage_percentage = self.calculate_coverage_percentage()

        min_distance_threshold = 50
        if coverage_percentage >= 90:
            min_distance_threshold = 50
        elif coverage_percentage >= 60:
            min_distance_threshold = 50

        valid_nodes = [
            node_index for node_index in valid_nodes
            if all(
                self.get_distance(self.get_node_position(node_index), pos) >= min_distance_threshold
                for pos in deployed_node_positions
            )
        ]

        valid_actions = list(valid_nodes) + [self.n_potential_nodes]  # 'do nothing'
        return valid_actions

    def _get_flattened_state(self):
        D_flat = self.state["D"].flatten()
        R_flat = self.state["R"].flatten()
        N_flat = self.state["N"].flatten()
        return np.concatenate([D_flat, R_flat, N_flat])

    def get_distance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def render(self, mode='human'):
        pass

    def total_deployed_nodes(self):
        return int(np.sum(self.state["D"]))

    def calculate_reward(self):
        alpha = 1
        beta = 0.001
        uncovered_area_percent = 100 - self.calculate_coverage_percentage()
        uncovered_area_penalty = uncovered_area_percent * alpha
        deployment_penalty = beta * self.total_deployed_nodes()
        base_reward = -uncovered_area_penalty - deployment_penalty

        coverage = self.calculate_coverage_percentage()
        deployed_nodes = self.total_deployed_nodes()
        coverage_reward = base_reward

        # 节点数惩罚
        if deployed_nodes > self.optimal_node_count:
            node_penalty = (deployed_nodes - self.optimal_node_count) * self.node_penalty_factor
        else:
            node_penalty = 0

        reward = coverage_reward - node_penalty
        return reward

    def update_coverage(self):
        for node_index, is_deployed in enumerate(self.state['D']):
            if is_deployed == 1:
                for (x, y) in self.coverage_map[node_index]:
                    self.coverage_grid[x, y] = 1

    def calculate_coverage_percentage(self):
        total_covered = np.sum(self.coverage_grid)
        total_area = self.grid_size * self.grid_size
        coverage_percentage = (total_covered / total_area) * 100
        return coverage_percentage

    def get_node_position(self, node_index):
        row = node_index // self.n_potential_nodes_per_row
        col = node_index % self.n_potential_nodes_per_row
        x = int(col * self.node_spacing * self.narrow)
        y = int(row * self.node_spacing * self.narrow)
        return x, y

    # --------------------------------------------------------------------------
    # 新增一个函数用于可视化Donor节点在地图中的位置
    def plot_donor_positions(self, filename="donor_positions.png"):
        """
        Plot donor positions among all potential node positions,
        saving the figure to filename.
        """
        plt.figure(figsize=(6,6))

        # 1. 画所有潜在节点 (蓝色小圆点)
        for node_idx in range(self.n_potential_nodes):
            x, y = self.get_node_position(node_idx)
            plt.plot(x, y, 'bo', markersize=2)

        # 2. 画 donor 节点 (红色叉号)
        for d_idx in self.donor_indices:
            x, y = self.get_node_position(d_idx)
            plt.plot(x, y, 'rx', markersize=8, label='Donor' if d_idx == self.donor_indices[0] else "")

        plt.title("Donor positions among potential node positions")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
    # --------------------------------------------------------------------------


# 全局参数
max_steps = 30
scale = 100
numberofdonor = 5

# 初始化环境、智能体并开始训练
env = NetworkDeploymentEnv()

# 先画一张 Donor 节点位置图
env.plot_donor_positions("my_donor_map.png")

state_dim = len(env.reset())
action_dim = env.action_space
agent = Agent(state_dim, action_dim)

# 训练
rewards = train(env, agent, episodes=20000)
