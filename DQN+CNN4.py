# new upadate on fixed donor and action reduction
# try fix donor position
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


class NetworkDeploymentEnv(gym.Env):
    def __init__(self):
        super(NetworkDeploymentEnv, self).__init__()
        self.coverage_map = None
        self.connections_map = None
        self.load_precomputed_data()
        self.map_size = 1000
        self.grid_size = 1000
        self.node_spacing = 50
        self.n_potential_nodes_per_row = self.grid_size // self.node_spacing
        self.n_potential_nodes = (self.grid_size // self.node_spacing) ** 2
        self.action_space = spaces.Discrete(self.n_potential_nodes + 1)
        self.coverage_grid = np.zeros((self.map_size, self.map_size), dtype=np.int8)
        self.observation_space = spaces.Dict({
            "D": spaces.MultiBinary((self.n_potential_nodes_per_row, self.n_potential_nodes_per_row)),
            "R": spaces.Box(low=0, high=20, shape=(self.n_potential_nodes_per_row, self.n_potential_nodes_per_row)),
            "N": spaces.Box(low=0, high=10, shape=(self.n_potential_nodes_per_row, self.n_potential_nodes_per_row))
        })
        self.overhead = 1.2
        self.node_data_rate = 2
        self.donor_data_rate = 15  # 15 Gbps
        self.coverage_radius = 200
        self.backhaul_radius = 300
        self.narrow = 1
        self.numberofdonor = numberofdonor
        self.current_step = 0
        self.max_steps = max_steps
        self.previous_actions = set()
        self.last_reward = None
        self.coverage_needs_update = True
        self.state_min = np.array([0, 0, 0])  # Minimum values for D, R, N channels
        self.state_max = np.array([1, 15, 10])  # Maximum values for D, R, N channels assuming these maxes
        self.donor_adjacency_matrices = np.zeros((self.numberofdonor, self.n_potential_nodes, self.n_potential_nodes))
        # self.donor_indices = self.get_fixed_donor_positions()
        self.reset()

    # def get_fixed_donor_positions(self):
    #     # Define fixed donor positions here (example: first few nodes in the grid)
    #     fixed_positions = [0, 10, 20, 30, 40]  # These are indices, modify as needed
    #     return fixed_positions

    def reset(self):
        self.state = {
            "D": np.zeros((self.n_potential_nodes_per_row, self.n_potential_nodes_per_row)),
            "R": np.zeros((self.n_potential_nodes_per_row, self.n_potential_nodes_per_row)),
            "N": np.zeros((self.n_potential_nodes_per_row, self.n_potential_nodes_per_row))}
        self.coverage_grid.fill(0)
        self.donor_adjacency_matrices.fill(0)
        self.donor_indices = np.random.choice(range(self.n_potential_nodes), self.numberofdonor, replace=False)
        for idx in self.donor_indices:
            x, y = divmod(idx, self.n_potential_nodes_per_row)
            self.state["D"][x, y] = 1
            self.state["R"][x, y] = self.donor_data_rate
            self.update_coverage_single_node(idx)
        self.current_step = 0
        self.last_reward = 0
        self.previous_actions = set()
        self.coverage_needs_update = True
        return self._get_cnn_compatible_state()

    def normalize_state(self, state):
        """ Normalize the given state to the range [0, 1]. """
        state_min_reshaped = self.state_min.reshape(3, 1, 1)
        state_max_reshaped = self.state_max.reshape(3, 1, 1)
        normalized_state = (state - state_min_reshaped) / (state_max_reshaped - state_min_reshaped)
        return normalized_state

    def _get_cnn_compatible_state(self):
        """ Get the current state and normalize it for CNN input. """
        D_channel = np.expand_dims(self.state["D"], axis=0)
        R_channel = np.expand_dims(self.state["R"], axis=0)
        N_channel = np.expand_dims(self.state["N"], axis=0)
        state = np.concatenate([D_channel, R_channel, N_channel], axis=0)
        return self.normalize_state(state)

    def load_precomputed_data(self):
        try:
            with open('coverage_map.pkl1002', 'rb') as f:
                self.coverage_map = pickle.load(f)
            with open('connections_map.pkl1002', 'rb') as f:
                self.connections_map = pickle.load(f)
        except FileNotFoundError:
            print("Failed to load the precomputed data files. Please check the files' existence and paths.")
            self.coverage_map = {}
            self.connections_map = {}
        except Exception as e:
            print(f"An error occurred while loading precomputed data: {e}")
            self.coverage_map = {}
            self.connections_map = {}

    def update_coverage_single_node(self, node_index):
        node_x, node_y = divmod(node_index, self.n_potential_nodes_per_row)
        if self.state["D"][node_x, node_y] == 1:
            for (x, y) in self.coverage_map.get(node_index, []):
                self.coverage_grid[x, y] = 1
        print('Added coverage:', np.sum(self.coverage_grid))

    def step(self, action_index):
        valid_actions = self.get_valid_actions()
        if action_index not in valid_actions:
            action_index = valid_actions[np.random.randint(len(valid_actions))]
        rewards = 0
        done = False
        node_index = action_index
        print(f"Action: deploy node at position {node_index}")
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
        if self.current_step >= self.max_steps:
            self.current_step = 0
            self.coverage_needs_update = True
            done = True
        elif self.calculate_coverage_percentage() >= 100:
            self.current_step = 0
            self.coverage_needs_update = True
            done = True
        return self._get_cnn_compatible_state(), rewards, done

    def deploy_node(self, node_index):
        coverage_before = np.sum(self.coverage_grid)
        print(f"Coverage before deployment: {coverage_before} grids")
        x, y = divmod(node_index, self.n_potential_nodes_per_row)
        if self.state["D"][x, y] == 1:
            self.state["D"][x, y] = 1
            rewards = self.calculate_reward()
            self.update_coverage()
            return rewards
        else:
            connected, donor_id = self.reconnect_node(node_index)
            if not connected:
                print(f"Deployed node could not find a node to connect. High penalty applied.")
                self.state["D"][x, y] = 0
                self.update_coverage()
                print(f"Failed to connect node at index {node_index}.")
                return self.calculate_reward()

            self.state["D"][x, y] = 1  # Deploy node
            self.update_coverage_single_node(node_index)
            coverage_after = np.sum(self.coverage_grid)
            coverage_increase = coverage_after - coverage_before
            # print(f"New coverage: {coverage_increase} additional grids.")
            # print(f"Successfully deployed and connected node at position {[x * self.node_spacing * self.narrow, y * self.node_spacing * self.narrow]}.")
            # print(f"Data rates: {self.state['R']}")
            # print("Data rate", np.sum(self.state['R']))
            # print("Node decision", np.sum(self.state['D']))
            # print("Connection decision:", np.sum(self.state['N']))
            return self.calculate_reward()

    def keep_node(self):
        return self.calculate_reward()

    def reconnect_node(self, node_index):
        for d_id, donor_index in enumerate(self.donor_indices):
            best_target = None
            max_data_rate = 2.4
            for target in self.connections_map.get(node_index, []):
                x, y = divmod(target, self.n_potential_nodes_per_row)
                if self.state["D"][x, y] == 1 and self.state["R"][x, y] > max_data_rate:
                    if target not in self.donor_indices or target == donor_index:
                        max_data_rate = self.state["R"][x, y]
                        best_target = target
            if best_target is not None:
                bx, by = divmod(best_target, self.n_potential_nodes_per_row)
                self.state["N"][bx, by] += 1
                self.donor_adjacency_matrices[d_id, best_target, node_index] = 1
                self.update_data_rates(d_id)
                return True, d_id
        return False, None

    def update_data_rates(self, donor_id):
        adjacency_matrix = self.donor_adjacency_matrices[donor_id]
        donor_index = self.donor_indices[donor_id]
        print('donor_index', donor_index)
        print('adjacency_matrix', np.sum(adjacency_matrix))
        # Use a set to keep track of nodes connected to the current donor
        connected_nodes = set()

        # Traverse the adjacency matrix to find all nodes connected to the current donor
        for node_index in range(self.n_potential_nodes):
            if adjacency_matrix[donor_index, node_index] > 0 or adjacency_matrix[node_index, donor_index] > 0 or np.sum(
                    adjacency_matrix[:, node_index]) > 0 or np.sum(adjacency_matrix[node_index, :]) > 0:
                connected_nodes.add(node_index)
        print('connected_nodes ', connected_nodes)
        # Calculate the shared data rate
        num_connected_nodes = len(connected_nodes)
        shared_data_rate = max(0, self.donor_data_rate - (num_connected_nodes * self.overhead * self.node_data_rate))

        # Update data rates for the current donor and its connected nodes
        for node_index in connected_nodes:
            x, y = divmod(node_index, self.n_potential_nodes_per_row)
            self.state["R"][x, y] = shared_data_rate

        print(f"Updated data rates for donor {donor_id} and its connected nodes.")

    # def update_data_rates(self, donor_id):
    #     adjacency_matrix = self.donor_adjacency_matrices[donor_id]
    #     print('adjacency_matrix',  np.sum(adjacency_matrix))
    #     num_connected_nodes = np.sum(adjacency_matrix)
    #     shared_data_rate = max(0, self.donor_data_rate - (num_connected_nodes * self.overhead * self.node_data_rate))

    #     for node_index in range(self.n_potential_nodes):
    #         if np.sum(adjacency_matrix[:, node_index]) > 0 or np.sum(adjacency_matrix[node_index, :]) > 0:
    #             x, y = divmod(node_index, self.n_potential_nodes_per_row)
    #             self.state["R"][x, y] = shared_data_rate
    # donor_index = self.donor_indices[donor_id]
    # dx, dy = divmod(donor_index, self.n_potential_nodes_per_row)
    # self.state["R"][dx, dy] = shared_data_rate

    # # Update donor data rate
    # donor_index = self.donor_indices[donor_id]
    # x, y = divmod(donor_index, self.n_potential_nodes_per_row)
    # self.state["R"][x, y] = shared_data_rate

    def calculate_distance_2d(self, x1, y1, x2, y2):
        """Calculate the Euclidean distance between two points in the grid."""
        dx = (x2 - x1) * self.node_spacing * self.narrow
        dy = (y2 - y1) * self.node_spacing * self.narrow
        return math.sqrt(dx ** 2 + dy ** 2)

    def render(self, mode='human'):
        pass

    def total_deployed_nodes(self):
        total_deployed_nodes = np.sum(self.state["D"])
        print("Total deployed nodes:", total_deployed_nodes)
        return total_deployed_nodes

    def calculate_reward(self):
        alpha = 100  # Penalty for uncovered area
        beta = 1  # Penalty for each deployed node

        total_area = self.grid_size * self.grid_size
        uncovered_area_percent = 100 - self.calculate_coverage_percentage()
        uncovered_area_penalty = uncovered_area_percent * alpha
        deployment_penalty = beta * self.total_deployed_nodes()

        reward = -uncovered_area_penalty - deployment_penalty
        return reward

    def get_node_position(self, node_index):
        row = node_index // (self.grid_size // self.node_spacing)
        col = node_index % (self.grid_size // self.node_spacing)
        x = col * self.node_spacing * self.narrow
        y = row * self.node_spacing * self.narrow
        return x, y

    def calculate_coverage_percentage(self):
        total_covered = np.sum(self.coverage_grid)
        total_area = self.grid_size * self.grid_size
        coverage_percentage = (total_covered / total_area) * 100
        print('Coverage percentage:', coverage_percentage)
        return coverage_percentage

    def update_coverage(self):
        for node_index, is_deployed in enumerate(self.state['D']):
            node_x, node_y = divmod(node_index, self.n_potential_nodes_per_row)
            if self.state["D"][node_x, node_y] == 1:
                for (x, y) in self.coverage_map.get(node_index, []):
                    self.coverage_grid[x, y] = 1
                print('Coverage increase:', np.sum(self.coverage_grid))

    def get_nodes_near_uncovered_areas(self, uncovered_positions, radius=200):
        valid_nodes = set()
        for x, y in uncovered_positions:
            for node_index in range(self.n_potential_nodes):
                node_x, node_y = self.get_node_position(node_index)
                distance = np.sqrt((node_x - x) ** 2 + (node_y - y) ** 2)
                if distance <= radius:
                    valid_nodes.add(node_index)
        return list(valid_nodes)

    def get_valid_actions(self):
        valid_nodes = set(range(self.n_potential_nodes))
        deployed_nodes = set(np.where(self.state["D"] == 1)[0])
        valid_nodes = valid_nodes - deployed_nodes

        # Get the positions of deployed nodes
        deployed_node_positions = [self.get_node_position(node_index) for node_index in deployed_nodes]

        # Calculate the current coverage percentage
        coverage_percentage = self.calculate_coverage_percentage()

        # Adjust the minimum distance threshold based on the coverage percentage
        min_distance_threshold = 200  # Initial threshold
        if coverage_percentage >= 100:
            min_distance_threshold = 100  # Relax the threshold when coverage is high
        elif coverage_percentage >= 60:
            min_distance_threshold = 100  # Intermediate threshold

        # Filter out nodes that are too close to previously deployed nodes
        valid_nodes = [node_index for node_index in valid_nodes
                       if all(self.get_distance(self.get_node_position(node_index), pos) >= min_distance_threshold
                              for pos in deployed_node_positions)]

        valid_actions = list(valid_nodes) + [self.n_potential_nodes]  # Include the 'do nothing' action
        return valid_actions

    def get_distance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def print_state_info(self):
        print('Current Step:', self.current_step)
        print('Deployed Nodes:')
        print(self.state["D"])
        print('Data Rates (R):')
        print(self.state["R"])
        print('Number of Connections (N):')
        print(self.state["N"])
        print('Coverage Percentage:', self.calculate_coverage_percentage())


# class DQN(nn.Module):
#     def __init__(self, input_shape, n_actions, dropout_rate=0.3):
#         super(DQN, self).__init__()
#         # First convolutional stream
#         self.features1 = nn.Sequential(
#             nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Modified stride/padding
#             nn.ReLU(inplace=True),
#         )
#         # Second convolutional stream
#         self.features2 = nn.Sequential(
#             nn.Conv2d(input_shape[0], 512, kernel_size=8, stride=4, padding=2),  # Adjusted to match features1
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # Adjusted to match features1
#             nn.ReLU(inplace=True),
#         )
#         self.fc_input_dim = self._get_conv_output(input_shape)

#         self.decision_maker = nn.Sequential(
#             nn.Linear(self.fc_input_dim, 2048),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(dropout_rate),
#             nn.Linear(2048, 1024),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(dropout_rate),
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(dropout_rate),
#             nn.Linear(512, n_actions)
#         )

#     def _get_conv_output(self, shape):
#         with torch.no_grad():
#             input = torch.rand(1, *shape)
#             output1 = self.features1(input)
#             output2 = self.features2(input)
#             # print(f"Output 1 shape: {output1.shape}")
#             # print(f"Output 2 shape: {output2.shape}")
#             output = torch.cat((output1, output2), dim=1)  # Concatenate along the channel dimension
#             return int(np.prod(output.size()[1:]))

#     def forward(self, x):
#         x1 = self.features1(x)
#         x2 = self.features2(x)
#         x = torch.cat((x1, x2), dim=1)  # Concatenate along the channel dimension
#         x = x.view(x.size(0), -1)
#         x = self.decision_maker(x)
#         return x
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, dropout_rate=0.3):
        super(DQN, self).__init__()
        self.features = nn.Sequential(
            # First block: Large kernels for capturing broad features
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second block: Medium kernels for intermediate features
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third block: Smaller kernels for detailed features
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Fourth block: Even smaller kernels to finalize feature extraction
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_input_dim = self._get_conv_output(input_shape)

        # Expanded fully connected layers
        self.decision_maker = nn.Sequential(
            nn.Linear(self.fc_input_dim, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_actions)
        )

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self.features(input)
            return int(np.prod(output.size()[1:]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.decision_maker(x)
        return x

# class DQN(nn.Module):
#     def __init__(self, input_shape, n_actions, dropout_rate=0.3):
#         super(DQN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=1, padding=3),
#             # Large kernel with stride for downsampling
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Medium kernel size
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Smaller kernel size
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.fc_input_dim = self._get_conv_output(input_shape)
#
#         self.decision_maker = nn.Sequential(
#             nn.Linear(self.fc_input_dim, 2048),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(dropout_rate),
#             nn.Linear(2048, 1024),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(dropout_rate),
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(dropout_rate),
#             nn.Linear(512, n_actions)
#         )
#
#     def _get_conv_output(self, shape):
#         with torch.no_grad():
#             input = torch.rand(1, *shape)
#             output = self.features(input)
#             return int(np.prod(output.size()[1:]))
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.decision_maker(x)
#         return x


class Agent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=80000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.gamma = gamma
        self.epsilon_start = epsilon_start  # Renamed from self.epsilon to self.epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay  # Use this for the decay calculation
        self.memory = deque(maxlen=5000)
        self.total_steps = 0
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.5,
                                                              min_lr=1e-6)

    def select_action(self, state,valid_actions):
        print('self.total_steps1', self.total_steps)
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.total_steps / self.epsilon_decay)
        if random.random() < eps_threshold:
            action_index = random.choice(valid_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state)
            q_values = q_values.squeeze().cpu().detach().numpy()
            valid_q_values = {action: q_values[action] for action in valid_actions}
            action_index = max(valid_q_values, key=valid_q_values.get)
        return action_index

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(device)  # Move to device
        actions = torch.LongTensor(actions).to(device)  # Move to device
        rewards = torch.FloatTensor(rewards).to(device)  # Move to device
        next_states = torch.FloatTensor(np.array(next_states)).to(device)  # Move to device
        dones = torch.FloatTensor(dones).to(device)
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = F.mse_loss(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step(loss)
        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())


def train(env, agent, episodes, batch_size=128, target_update=64, scale=100):
    episode_rewards = []
    episode_losses = []
    data = {
        "Episode": [],
        "Total Reward": [],
        "Avg Loss": [],
        "Deployed Nodes": [],
        "Coverage": []
    }
    all_steps_details = []  # To store details of all steps

    for episode in range(episodes):
        start_time = time.time()
        state = env.reset()
        total_reward = 0
        total_loss = 0
        done = False
        step_count = 0

        while not done:
            action_start_time = time.time()
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            step_start_time = time.time()
            next_state, reward, done = env.step(action)
            step_duration = time.time() - step_start_time

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

            if step_count >= env.max_steps:
                done = True
            elif env.calculate_coverage_percentage() >= 100:
                done = True

        episode_duration = time.time() - start_time
        avg_loss = total_loss / step_count
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        deployed_nodes = env.total_deployed_nodes()
        coverage_percentage = env.calculate_coverage_percentage()
        data["Episode"].append(episode)
        data["Total Reward"].append(total_reward)
        data["Avg Loss"].append(avg_loss)
        data["Deployed Nodes"].append(deployed_nodes)
        data["Coverage"].append(coverage_percentage)

    avg_rewards_per_scale_episodes = [np.mean(episode_rewards[i:i + scale]) for i in
                                      range(0, len(episode_rewards), scale)]
    avg_losses_per_scale_episodes = [np.mean(episode_losses[i:i + scale]) for i in range(0, len(episode_losses), scale)]
    avg_numofnodes_per_scale_episodes = [np.mean(data["Deployed Nodes"][i:i + scale]) for i in
                                         range(0, len(data["Deployed Nodes"]), scale)]
    avg_coverage_per_scale_episodes = [np.mean(data["Coverage"][i:i + scale]) for i in
                                       range(0, len(data["Coverage"]), scale)]
    episodes_scale = list(range(0, len(episode_rewards), scale))

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'all_episodes_step_details_{timestamp}.csv'
    with open(filename, mode='w', newline='') as file:
        fieldnames = ["Episode", "Step", "Action", "Reward", "Coverage"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for detail in all_steps_details:
            writer.writerow(detail)

    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Reward per Episode')
    plt.plot(episodes_scale, avg_rewards_per_scale_episodes, label='Avg Reward per 100 Episodes', color='red',
             linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Episode vs Reward')
    filename = f'Episode_vs_Reward_{timestamp}.png'
    plt.savefig(filename)
    plt.close()

    plt.plot(episode_losses, label='Loss per Episode')
    plt.plot(episodes_scale, avg_losses_per_scale_episodes, label='Avg Loss per 100 Episodes', color='blue',
             linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Episode vs Loss')
    filename = f'Episode_vs_Loss_{timestamp}.png'
    plt.savefig(filename)
    plt.close()

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
    plt.plot(episodes_scale, avg_coverage_per_scale_episodes, label='Avg nodes per 100 Episodes', color='red',
             linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Coverage Percentage')
    plt.title('Episode vs Coverage Percentage')
    plt.legend()

    plt.tight_layout()
    filename = f'Episode_vs_Coverage_Percentage_{timestamp}.png'
    plt.savefig(filename)
    plt.close()

    filename = f'training_data_{timestamp}.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Reward', 'Avg Loss', 'Deployed Nodes', 'Coverage'])
        for i in range(len(data['Episode'])):
            writer.writerow(
                [data['Episode'][i], data['Total Reward'][i], data['Avg Loss'][i], data['Deployed Nodes'][i],
                 data['Coverage'][i]])

    return episode_rewards, episode_losses


# Define parameters
max_steps = 100
scale = 100
numberofdonor = 5
env = NetworkDeploymentEnv()
n_potential_nodes_per_row = env.n_potential_nodes_per_row
state_dim = (3, n_potential_nodes_per_row, n_potential_nodes_per_row)
action_dim = env.action_space.n
agent = Agent(state_dim, action_dim)
rewards = train(env, agent, episodes=5000)
