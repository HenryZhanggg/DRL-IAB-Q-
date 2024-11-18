import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt  # Add this line
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import time
import copy
import csv
import datetime
import os

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NetworkDeploymentEnv(gym.Env):
    """
    Custom Environment for Network Deployment compatible with OpenAI Gym interface.
    This environment allows deploying nodes on a grid to maximize coverage while ensuring connectivity to donor nodes.
    Donor nodes and blocked nodes remain fixed throughout the training process.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 map_size=1000,
                 grid_size=1000,
                 node_spacing=50,
                 number_of_donors=5,
                 blocked_percentage=0.1,
                 coverage_threshold=95,
                 optimal_node_count=22,
                 node_penalty_factor=100,
                 max_steps=50,
                 coverage_radius=200,
                 backhaul_radius=300):
        super(NetworkDeploymentEnv, self).__init__()

        # Environment Parameters
        self.overhead = 1.2
        self.node_data_rate = 2
        self.donor_data_rate = 50  # 15 Gbps
        self.map_size = map_size
        self.grid_size = grid_size
        self.node_spacing = node_spacing
        self.n_potential_nodes_per_row = int(self.grid_size // self.node_spacing)
        self.n_potential_nodes = self.n_potential_nodes_per_row ** 2
        self.number_of_donors = number_of_donors
        self.coverage_threshold = coverage_threshold
        self.optimal_node_count = optimal_node_count
        self.node_penalty_factor = node_penalty_factor
        self.max_steps = max_steps
        self.coverage_radius = coverage_radius
        self.backhaul_radius = backhaul_radius

        # Action Space: Deploy at any node or do nothing
        self.action_space = spaces.Discrete(self.n_potential_nodes + 1)  # +1 for 'do nothing'
        self.previous_actions = set()
        self.donor_adjacency_matrices = np.zeros(
            (self.number_of_donors, self.n_potential_nodes, self.n_potential_nodes))

        # Observation Space
        self.observation_space = spaces.Dict({
            "D": spaces.MultiBinary(self.n_potential_nodes),  # Deployment status of nodes
            "R": spaces.Box(low=0, high=50, shape=(self.n_potential_nodes,), dtype=np.float32),  # Remaining resources
            "N": spaces.Box(low=0, high=10, shape=(self.n_potential_nodes,), dtype=np.float32)  # Number of connections
        })

        # Initialize state variables
        self.state = {
            "D": np.zeros(self.n_potential_nodes, dtype=np.int8),
            "R": np.zeros(self.n_potential_nodes, dtype=np.float32),
            "N": np.zeros(self.n_potential_nodes, dtype=np.float32)
        }

        # Initialize blocked grids and nodes
        self.blocked_percentage = blocked_percentage  # 确保 blocked_percentage 可用
        self.blocked_nodes = self.get_fixed_blocked_nodes(self.blocked_percentage)

        # Coverage and Blocked Grids
        self.coverage_grid = np.zeros((self.map_size, self.map_size), dtype=np.int8)
        self.blocked_grid = np.zeros((self.map_size, self.map_size), dtype=np.int8)
        self.coverage_map = {}
        self.connections_map = {}
        self.generate_or_load_precomputed_data()
        # self.initialize_blocked_grid()

        # Initialize donor nodes
        self.donor_indices = self.get_fixed_donor_positions()
        for idx in self.donor_indices:
            self.state["D"][idx] = 1  # Mark as deployed
            self.state["R"][idx] = 50  # Assign maximum resources to donors
            self.update_coverage_single_node(idx)

        total_initial_covered = np.sum(self.coverage_grid)
        total_area = self.map_size * self.map_size
        percentage_covered = (total_initial_covered / total_area) * 100
        print(f"Total11111: {total_initial_covered}")
        print(f"Percentag22222: {percentage_covered:.2f}%")
        # Step Counter
        self.current_step = 0
        self.previous_coverage = 0.0  # To store the previous coverage percentage

    def get_neighboring_nodes(self, node_index, radius=1):
        """
        获取指定节点周围指定半径范围内的所有邻近节点索引。

        :param node_index: 节点的索引
        :param radius: 邻近范围的半径（以网格单位为单位）
        :return: 邻近节点的索引列表
        """
        neighbors = []
        row = node_index // self.n_potential_nodes_per_row
        col = node_index % self.n_potential_nodes_per_row

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue  # 跳过自身
                neighbor_row = row + dr
                neighbor_col = col + dc
                if 0 <= neighbor_row < self.n_potential_nodes_per_row and 0 <= neighbor_col < self.n_potential_nodes_per_row:
                    neighbor_index = neighbor_row * self.n_potential_nodes_per_row + neighbor_col
                    neighbors.append(neighbor_index)
        return neighbors

    def get_fixed_donor_positions(self):

        center_index = (self.n_potential_nodes - 1) // 2
        donors_per_row = self.n_potential_nodes_per_row

        # Radius for donor placement (number of nodes away from center)
        donor_radius = donors_per_row // 4

        donor_positions = []
        star_angles = [0, 144, 288, 72, 216]  # Angles for a 5-star shape in degrees

        center_row = center_index // self.n_potential_nodes_per_row
        center_col = center_index % self.n_potential_nodes_per_row

        for angle in star_angles:
            radian = math.radians(angle)
            # Calculate offset in grid
            row_offset = int(donor_radius * math.sin(radian))
            col_offset = int(donor_radius * math.cos(radian))
            donor_row = center_row + row_offset
            donor_col = center_col + col_offset

            # Ensure donor positions are within grid bounds
            donor_row = max(0, min(self.n_potential_nodes_per_row - 1, donor_row))
            donor_col = max(0, min(self.n_potential_nodes_per_row - 1, donor_col))

            donor_index = donor_row * self.n_potential_nodes_per_row + donor_col
            donor_positions.append(donor_index)

        return list(set(donor_positions))
    def get_fixed_blocked_nodes(self, blocked_percentage):
        """
        Loads blocked nodes from a file if available; otherwise, generates new blocked nodes.
        """
        try:
            with open('blocked_nodes.pkl', 'rb') as f:
                blocked_nodes = pickle.load(f)
            print("Blocked nodes loaded from file.")
            return blocked_nodes
        except FileNotFoundError:
            print("Blocked nodes file not found. Generating new blocked nodes.")
            num_blocked_nodes = int(blocked_percentage * self.n_potential_nodes)
            # Ensure blocked nodes do not include donor nodes
            potential_blocked = list(set(range(self.n_potential_nodes)) - set(self.donor_indices))
            if num_blocked_nodes > len(potential_blocked):
                num_blocked_nodes = len(potential_blocked)
            blocked_nodes = set(np.random.choice(potential_blocked, num_blocked_nodes, replace=False))
            return blocked_nodes

    def generate_or_load_precomputed_data(self):
        coverage_map_file = 'coverage_map.pkl1002'
        connections_map_file = 'connections_map.pkl1002'

        if os.path.exists(coverage_map_file) and os.path.exists(connections_map_file):
            try:
                with open(coverage_map_file, 'rb') as f:
                    self.coverage_map = pickle.load(f)
                with open(connections_map_file, 'rb') as f:
                    self.connections_map = pickle.load(f)
                print("Precomputed coverage and connections maps loaded.")
            except Exception as e:
                print(f"Error loading precomputed data: {e}")
                # Handle the exception as needed
        else:
            print("Precomputed data files not found. Please generate them before running the environment.")
            # Optionally, you can call a method to generate the data here
            # self.generate_dummy_precomputed_data()

    def initialize_blocked_grid(self):
        """
        Initializes the blocked grid based on blocked node positions.
        Blocked nodes represent areas that do not require coverage.
        Additionally, marks a surrounding 10m radius area around each blocked node as already covered.
        """
        for node_index in self.blocked_nodes:
            # Mark the blocked node itself
            coverage_areas = self.coverage_map.get(node_index, [])
            for (x, y) in coverage_areas:
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    self.blocked_grid[y, x] = 1  # 标记为阻塞区域

            # Mark areas within 10 meters as already covered
            node_x, node_y = self.get_node_position(node_index)
            # Determine the range in the grid to consider
            x_start = max(0, node_x - 10)
            x_end = min(self.map_size, node_x + 10 + 1)
            y_start = max(0, node_y - 10)
            y_end = min(self.map_size, node_y + 10 + 1)

            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    distance = np.sqrt((x - node_x) ** 2 + (y - node_y) ** 2)
                    if distance <= 10:  # 10 米半径
                        if self.blocked_grid[y, x] == 0:
                            self.coverage_grid[y, x] = 1  # 直接在 coverage_grid 中标记为已覆盖

        # 计算初始已覆盖的网格数量和百分比
        total_initial_covered = np.sum(self.coverage_grid)
        total_area = self.map_size * self.map_size
        percentage_covered = (total_initial_covered / total_area) * 100
        print(f"Total grids initially covered due to blocked nodes and their surrounding areas: {total_initial_covered}")
        print(f"Percentage of total area already covered: {percentage_covered:.6f}%")

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.state = {
            "D": np.zeros(self.n_potential_nodes, dtype=np.int8),
            "R": np.zeros(self.n_potential_nodes, dtype=np.float32),
            "N": np.zeros(self.n_potential_nodes, dtype=np.float32)
        }
        self.state["N"] = np.zeros((self.n_potential_nodes))
        self.state["R"] = np.zeros(self.n_potential_nodes)
        self.coverage_grid.fill(0)
        # Re-deploy donor nodes
        for idx in self.donor_indices:
            self.state["D"][idx] = 1
            self.state["R"][idx] = 50
            self.update_coverage_single_node(idx)

        # Reset step counter
        self.current_step = 0
        self.previous_coverage = self.calculate_coverage_percentage()
        self.donor_adjacency_matrices = np.zeros(
            (self.number_of_donors, self.n_potential_nodes, self.n_potential_nodes))
        self.previous_actions = set()  # 也可以选择重置之前的动作集合
        self.last_reward = 0
        self.coverage_needs_update = True
        return self._get_flattened_state()

    def step(self, action_index):
        """
        Executes the given action in the environment and returns the next state, reward, and done flag.
        """
        # Store coverage before the action
        coverage_before = self.calculate_coverage_percentage()

        # Get valid actions
        valid_actions = self.get_valid_actions()
        if action_index not in valid_actions:
            action_index = random.choice(valid_actions)

        rewards = 0
        done = False
        node_index = action_index

        # Action execution
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

        # Check for coverage improvement
        coverage_after = self.calculate_coverage_percentage()
        if coverage_after <= coverage_before and coverage_after < self.coverage_threshold:
            penalty = -500  # Adjust this penalty as needed
            rewards += penalty

        # Check for episode termination
        if self.current_step >= self.max_steps:
            done = True
        elif coverage_after >= self.coverage_threshold:
            done = True

        # Get next state
        next_state = self._get_flattened_state()

        # **Added Print Statements**
        # action_position = self.get_node_position(node_index) if node_index < self.n_potential_nodes else "Do Nothing"
        # print(f"Action Taken: {'Deploy Node' if node_index < self.n_potential_nodes else 'Do Nothing'}")
        # print(f"Node Index: {node_index}")
        # print(f"Node Position: {action_position}")
        # print(f"Reward Received: {rewards}")
        # print(f"Coverage Before Action: {coverage_before:.2f}%")
        # print(f"Coverage After Action: {coverage_after:.2f}%\n")

        return next_state, rewards, done

    def deploy_node(self, node_index):
        node_x, node_y = self.get_node_position(node_index)
        # print(f"Attempting to deploy node at position: ({node_x}, {node_y})")
        coverage_before = np.sum(self.coverage_grid)
        # print(f"Coverage before deployment: {coverage_before} grids")
        if self.state["D"][node_index] == 1:
            self.state["D"][node_index] = 1
            self.update_coverage()
            # print(f"Node {node_index} at position ({node_x}, {node_y}) is already deployed.")
            return self.calculate_reward()
        else:
            connected, donor_id = self.reconnect_node(node_index)
            if not connected:
                self.state["D"][node_index] = 0
                # print(f"Failed to connect node at index {node_index}.")
                return self.calculate_reward()
            self.state["D"][node_index] = 1
            self.update_coverage_single_node(node_index)
            coverage_after = np.sum(self.coverage_grid)
            coverage_increase = coverage_after - coverage_before
            # print(f"Node deployed at ({node_x}, {node_y}). New coverage: {coverage_increase} additional grids.")
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
            if adjacency_matrix[donor_index, node_index] > 0 or adjacency_matrix[node_index, donor_index] > 0 or np.sum(
                    adjacency_matrix[:, node_index]) > 0 or np.sum(adjacency_matrix[node_index, :]) > 0:
                connected_nodes.add(node_index)

        num_connected_nodes = len(connected_nodes)
        shared_data_rate = max(0, self.donor_data_rate - (num_connected_nodes * self.overhead * self.node_data_rate))

        for node_index in connected_nodes:
            self.state["R"][node_index] = shared_data_rate
        # print(f"Updated data rates for donor {donor_id} and its connected nodes.")
        return True

    def calculate_reward(self):
        alpha = 1
        beta = 0.0001
        uncovered_area_percent = 100 - self.calculate_coverage_percentage()
        uncovered_area_penalty = uncovered_area_percent * alpha
        deployment_penalty = beta * self.total_deployed_nodes()
        base_reward = -uncovered_area_penalty - deployment_penalty
        coverage = self.calculate_coverage_percentage()
        deployed_nodes = self.total_deployed_nodes()
        coverage_reward = base_reward
        # if coverage < 95:
        #     coverage_reward += (base_reward - (coverage / 95) * 10)  # Small linear reward below 95%
        # elif 95 <= coverage < self.coverage_threshold:
        #     # Exponential reward between 95% and 98%
        #     coverage_reward += (base_reward + math.exp(coverage - 95))

        # Penalty for excessive nodes
        if deployed_nodes > self.optimal_node_count:
            node_penalty = (deployed_nodes - self.optimal_node_count) * self.node_penalty_factor
        else:
            node_penalty = 0
        reward = coverage_reward - node_penalty
        # print('reward',reward)
        return reward

    def get_valid_actions(self):
        """
        Returns a list of valid actions (node indices that can be deployed or 'do nothing'),
        excluding nodes that are blocked or too close to already deployed nodes.
        """
        # Get the set of all potential node indices
        all_nodes = set(range(self.n_potential_nodes))
        # Get the set of deployed nodes
        deployed_nodes = set(np.where(self.state["D"] == 1)[0])
        # Get the set of blocked nodes
        blocked_nodes = self.blocked_nodes  # Ensure self.blocked_nodes is defined and is a set
        # Valid nodes are those that are not deployed and not blocked
        valid_nodes = all_nodes - deployed_nodes - blocked_nodes

        # Get positions of deployed nodes
        deployed_node_positions = [self.get_node_position(node_index) for node_index in deployed_nodes]

        # Minimum distance threshold to avoid deploying nodes too close to each other
        min_distance_threshold = 100   # Adjust this value as needed

        # Filter out nodes that are too close to deployed nodes
        valid_nodes = [node_index for node_index in valid_nodes
                       if all(self.get_distance(self.get_node_position(node_index), pos) >= min_distance_threshold
                              for pos in deployed_node_positions)]

        # Include 'do nothing' action
        valid_actions = list(valid_nodes) + [self.n_potential_nodes]
        return valid_actions

    def _get_flattened_state(self):
        """
        Returns the current state as a flattened numpy array.
        """
        D_flat = self.state["D"].flatten()
        R_flat = self.state["R"].flatten()
        N_flat = self.state["N"].flatten()
        flattened_state = np.concatenate([D_flat, R_flat, N_flat])
        return flattened_state

    def get_node_position(self, node_index):
        """
        Returns the (x, y) position of a node based on its index.
        The position is adjusted to match the coverage grid coordinate system.
        """
        row = node_index // self.n_potential_nodes_per_row
        col = node_index % self.n_potential_nodes_per_row
        x = col * self.node_spacing
        y = row * self.node_spacing
        return int(x), int(y)  # Convert to integers for grid indexing

    def update_coverage_single_node(self, node_index):
        """
        Updates the coverage grid based on the coverage map of a deployed node.
        """
        if self.state["D"][node_index] == 1:
            for (x, y) in self.coverage_map.get(node_index, []):
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    # Do not mark blocked areas as covered
                    if self.blocked_grid[x, y] == 0:
                        self.coverage_grid[x, y] = 1

    def calculate_coverage_percentage(self):
        """
        Calculates the percentage of the map that is covered, excluding blocked areas.
        """
        total_covered = np.sum(self.coverage_grid)
        total_area = self.map_size * self.map_size
        blocked_area = np.sum(self.blocked_grid)
        required_coverage_area = total_area - blocked_area
        coverage_percentage = (total_covered / required_coverage_area) * 100 if required_coverage_area > 0 else 0
        return coverage_percentage


    def total_deployed_nodes(self):
        """
        Returns the total number of deployed nodes.
        """
        return int(np.sum(self.state["D"]))

    def get_distance(self, pos1, pos2):
        """
        Calculates Euclidean distance between two positions.
        """
        x1, y1 = pos1
        x2, y2 = pos2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def close(self):
        """
        Clean up the environment.
        """
        pass


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.fc4 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x


class Agent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=80000, pretrained=False, model_path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=80000)
        self.total_steps = 0
        self.taken_actions = set()  # Initialize set to track actions taken in the current episode
        self.reward_mean = 0
        self.reward_std = 1
        self.alpha = 0.01  # Smoothing factor for running mean and std
        self.pretrained = pretrained
        self.model_path = model_path

        if self.pretrained and self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)  # 加载预训练模型
            print(f"已加载预训练模型：{self.model_path}")
        else:
            print("未找到预训练模型，开始从头训练。")

    def load_model(self, path):
        """
        从指定路径加载模型的状态字典。
        """
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"模型已从 {path} 加载。")

    def select_action(self, state, valid_actions):
        """
        Selects an action using epsilon-greedy strategy, considering only valid actions.
        """
        state = self.normalize_state(state)
        valid_actions = [action for action in valid_actions if action not in self.taken_actions]

        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.total_steps / self.epsilon_decay)
        if random.random() < eps_threshold or not valid_actions:
            # Ensure there's always at least one valid action
            action_index = random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            q_values = q_values.squeeze().cpu().numpy()
            # Select the action with the highest Q-value among valid actions
            valid_q_values = q_values[valid_actions]
            max_q_index = np.argmax(valid_q_values)
            action_index = valid_actions[max_q_index]

        self.taken_actions.add(action_index)  # Record action as taken
        return action_index

    def reset_actions(self):
        """
        Resets the taken actions set at the beginning of each episode.
        """
        self.taken_actions.clear()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay memory.
        """
        reward = self.normalize_reward(reward)
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self, batch_size):
        """
        Samples a batch of transitions and performs a learning step.
        """
        if len(self.memory) < batch_size:
            return 0
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states, dtype=np.float32), device=device, dtype=torch.float32)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Double DQN: Select action using online network, evaluate with target network
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1]
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update learning rate scheduler
        # self.scheduler.step(loss)

        return loss.item()

    def update_target_network(self):
        """
        Updates the target network to match the online network.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def normalize_state(self, state):
        """
        Normalizes the state to have values between 0 and 1.
        """
        state = np.array(state)
        return (state - state.min()) / (state.max() - state.min() + 1e-5)

    def normalize_reward(self, reward):
        # Optionally adjust or remove normalization
        self.reward_mean = self.alpha * reward + (1 - self.alpha) * self.reward_mean
        self.reward_std = self.alpha * (reward - self.reward_mean) ** 2 + (1 - self.alpha) * self.reward_std
        # Clip the reward to prevent extreme values
        reward_clipped = np.clip(reward, -10, 10)
        #return (reward_clipped - self.reward_mean) / (np.sqrt(self.reward_std) + 1e-5)

        return reward

    def get_epsilon(self):
        """
        Returns the current value of epsilon.
        """
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.total_steps / self.epsilon_decay)

    def get_current_lr(self):
        """
        Returns the current learning rate.
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


def train(env, agent, episodes, batch_size=512, target_update=128, scale=300, save_model_interval=20000, pretrained_model_path=None):
    episode_rewards = []
    episode_losses = []
    episode_coverages = []
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

        end_episode_time = time.time()

        avg_loss = total_loss / step_count if step_count > 0 else 0
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        deployed_nodes = env.total_deployed_nodes()
        coverage_percentage = env.calculate_coverage_percentage()
        episode_coverages.append(coverage_percentage)
        epsilon = agent.get_epsilon()
        current_lr = agent.get_current_lr()

        data["Episode"].append(episode)
        data["Total Reward"].append(total_reward)
        data["Avg Loss"].append(avg_loss)
        data["Deployed Nodes"].append(deployed_nodes)
        data["Coverage"].append(coverage_percentage)

        # Save model periodically
        if (episode + 1) % save_model_interval == 0 and (episode + 1) != 0:
            torch.save(agent.model.state_dict(), f'model_episode_{episode + 1}.pth')
            print(f'Model saved at episode {episode + 1}')

        # Print results after each episode
        print(f'End of Episode {episode + 1}:')
        print(f'Total Reward: {total_reward}')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Total Deployed Nodes: {deployed_nodes}')
        print(f'Coverage: {coverage_percentage:.2f}%')
        print(f'Epsilon: {epsilon:.4f}')
        print(f'Learning Rate: {current_lr:.6f}\n')

    # Save final model
    torch.save(agent.model.state_dict(), '1000-3.pth')
    print('Final model saved.')

    # Save all step details to CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'all_episodes_step_details_{timestamp}.csv'
    with open(filename, mode='w', newline='') as file:
        fieldnames = ["Episode", "Step", "Action", "Reward", "Coverage"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for detail in all_steps_details:
            writer.writerow(detail)

    # Save training data to CSV
    filename = f'training_data_{timestamp}.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Reward', 'Avg Loss', 'Deployed Nodes', 'Coverage'])
        for i in range(len(data['Episode'])):
            writer.writerow(
                [data['Episode'][i], data['Total Reward'][i], data['Avg Loss'][i], data['Deployed Nodes'][i],
                 data['Coverage'][i]])

    return episode_rewards, episode_losses, episode_coverages  # Return coverages


def main():
    # Parameters
    max_steps = 50
    scale = 100
    number_of_donors = 5
    episodes = 20000
    batch_size = 512
    target_update = 128
    save_model_interval = 20000

    # Transfer Learning Parameters
    use_transfer_learning = False  # 设置为 True 以使用迁移学习
    pretrained_model_info = {
        'path': 'model_final_DQN.pth'  # 预训练模型的路径
    } if use_transfer_learning else None

    # Initialize environment (1000x1000 environment)
    env = NetworkDeploymentEnv(
        map_size=1000,
        grid_size=1000,
        node_spacing=50,
        number_of_donors=number_of_donors,
        blocked_percentage=0.1,
        coverage_threshold=100,  # Adjusted to 100%
        optimal_node_count=22,
        node_penalty_factor=100,
        max_steps=max_steps,
        coverage_radius=200,
        backhaul_radius=300
    )

    # Reset environment to get the initial state
    initial_state = env.reset()

    # Get state and action dimensions
    state_dim = len(initial_state)
    action_dim = env.action_space.n

    # Initialize agent with transfer learning parameters
    agent = Agent(
        state_dim,
        action_dim,
        pretrained=use_transfer_learning and pretrained_model_info is not None,
        model_path=pretrained_model_info['path'] if use_transfer_learning else None
    )

    # Start training
    rewards, losses, coverages = train(
        env,
        agent,
        episodes=episodes,
        batch_size=batch_size,
        target_update=target_update,
        save_model_interval=save_model_interval,
        pretrained_model_path=pretrained_model_info
    )

    # Optionally, save rewards and losses for later analysis
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Plot and save the results
    plot_and_save_results(rewards, losses, coverages, timestamp)


def moving_average(data, window_size=500):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_and_save_results(rewards, losses, coverages, timestamp):
    """
    Plots and saves the rewards, losses, and coverage percentages over episodes with a moving average.
    """
    # Plot total rewards per episode with moving average
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Rewards")
    plt.plot(moving_average(rewards), label=f"Moving Average (window=500)", color='orange')
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'total_rewards_{timestamp}.png')
    plt.close()

    # Plot average losses per episode with moving average
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Losses")
    plt.plot(moving_average(losses), label=f"Moving Average (window=500)", color='orange')
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'average_losses_{timestamp}.png')
    plt.close()

    # Plot coverage percentages per episode with moving average
    plt.figure(figsize=(10, 6))
    plt.plot(coverages, label="Coverage")
    plt.plot(moving_average(coverages), label=f"Moving Average (window=500)", color='orange')
    plt.title('Coverage Percentage per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Coverage (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'coverage_percentages_{timestamp}.png')
    plt.close()



if __name__ == "__main__":
    main()
