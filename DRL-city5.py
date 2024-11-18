import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import copy
import csv
import datetime
import os
import pandas as pd

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
                 coverage_threshold=100,
                 optimal_node_count=26,
                 node_penalty_factor=5,
                 max_steps=250,
                 coverage_radius=200,
                 backhaul_radius=300):
        super(NetworkDeploymentEnv, self).__init__()

        # Environment Parameters
        self.overhead = 1.2
        self.node_data_rate = 2
        self.donor_data_rate = 50  # 50 Gbps
        self.map_size = map_size
        self.coverage_threshold = coverage_threshold
        self.optimal_node_count = optimal_node_count
        self.node_penalty_factor = node_penalty_factor
        self.max_steps = max_steps
        self.coverage_radius = coverage_radius
        self.backhaul_radius = backhaul_radius
        self.penalty =0
        # Load potential node places
        self.load_potential_node_places()

        # Action Space: Deploy at any node or do nothing
        self.action_space = spaces.Discrete(self.n_potential_nodes + 1)  # +1 for 'do nothing'
        self.previous_actions = set()

        # Observation Space
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_potential_nodes * 3,), dtype=np.float32)

        # Initialize state variables
        self.state = {
            "D": np.zeros(self.n_potential_nodes, dtype=np.int8),
            "R": np.zeros(self.n_potential_nodes, dtype=np.float32),
            "N": np.zeros(self.n_potential_nodes, dtype=np.float32)
        }

        self.donor_indices = self.get_fixed_donor_positions()
        for idx in self.donor_indices:
            self.state["D"][idx] = 1  # Mark as deployed
            self.state["R"][idx] = 50  # Assign maximum resources to donors

        # Coverage and Blocked Grids
    # Coverage and Blocked Grids Initialization
        self.coverage_grid = np.zeros((self.map_size, self.map_size), dtype=np.int8)  # Initialize the coverage grid
        self.blocked_grid = np.zeros((self.map_size, self.map_size), dtype=np.int8)  # Initialize the blocked grid
        # Load precomputed data
        self.initialize_blocked_grid()

        self.generate_or_load_precomputed_data()

        # Initialize coverage based on donor nodes
        for idx in self.donor_indices:
            self.update_coverage_single_node(idx)

        # Step Counter
        self.current_step = 0
        self.previous_coverage = 0.0  # To store the previous coverage percentage
        self.donor_adjacency_matrices = []
        for donor_index in self.donor_indices:
            adjacency_matrix = np.zeros((self.n_potential_nodes, self.n_potential_nodes), dtype=np.int8)
            self.donor_adjacency_matrices.append(adjacency_matrix)
        self.donor_adjacency_matrices = np.array(self.donor_adjacency_matrices)


    def load_potential_node_places(self):
        """
        Loads potential node places from 'potential_node_places.csv' and updates related attributes.
        """
        nodes_df = pd.read_csv('potential_node_places_modified.csv')
        self.nodes_df = nodes_df
        self.node_positions = nodes_df[['x', 'y']].values  # Numpy array of positions
        self.n_potential_nodes = len(nodes_df)

        # Identify donor nodes
        self.donor_indices = nodes_df[nodes_df['type'] == 'Donor'].index.tolist()
        print(f"Potential node places loaded. Total nodes: {self.n_potential_nodes}, Donor nodes: {len(self.donor_indices)}")

    def get_fixed_donor_positions(self):
        """
        Loads donor positions from the provided donor node data.
        """
        donor_nodes = pd.read_csv('potential_node_places_modified.csv')
        donor_nodes = donor_nodes[donor_nodes['type'] == 'Donor']
        donor_indices = donor_nodes.index.tolist()
        return donor_indices

    def generate_or_load_precomputed_data(self):
        coverage_map_file = 'coverage_map_modified.pkl'
        connections_map_file = 'connections_map_modified.pkl'

        if os.path.exists(coverage_map_file) and os.path.exists(connections_map_file):
            try:
                with open(coverage_map_file, 'rb') as f:
                    self.coverage_map = pickle.load(f)
                with open(connections_map_file, 'rb') as f:
                    self.connections_map = pickle.load(f)
                print("Precomputed coverage and connections maps loaded.")
            except Exception as e:
                raise RuntimeError(f"Error loading precomputed data: {e}")
        else:
            raise FileNotFoundError("Precomputed data files not found. Please generate them before running the environment.")

    def initialize_blocked_grid(self):
        """
        Initializes the blocked grid based on the building grid data.
        """
        try:
            with open('building_grid_modified.pkl', 'rb') as f:
                self.blocked_grid = pickle.load(f)
            print("Building grid data loaded for blocked areas.")
        except FileNotFoundError:
            raise FileNotFoundError("Building grid data not found. Please generate 'building_grid.pkl' before running the environment.")

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.state = {
            "D": np.zeros(self.n_potential_nodes, dtype=np.int8),
            "R": np.zeros(self.n_potential_nodes, dtype=np.float32),
            "N": np.zeros(self.n_potential_nodes, dtype=np.float32)
        }
        self.coverage_grid.fill(0)
        # coverage_grid_file = 'coverage_map_modified.pkl'
        # if os.path.exists(coverage_grid_file):
        #     with open(coverage_grid_file, 'rb') as f:
        #         self.coverage_grid = pickle.load(f)

        block_grid_file = 'building_grid_modified.pkl'
        if os.path.exists(block_grid_file):
            with open(block_grid_file, 'rb') as f:
                self.blocked_grid = pickle.load(f)
        # print('initial_blocked_grids',np.sum(self.blocked_grid))
        # Re-deploy donor nodes
        for idx in self.donor_indices:
            self.state["D"][idx] = 1
            self.state["R"][idx] = 50
            self.update_coverage_single_node(idx)
        # print('initial_coverage_grids',np.sum(self.coverage_grid))
        # Reset step counter
        self.current_step = 0
        self.previous_coverage = self.calculate_coverage_percentage()
        self.donor_adjacency_matrices = np.zeros(
            (5, self.n_potential_nodes, self.n_potential_nodes))
        # Reset any other necessary variables
        self.previous_actions = set()
        self.last_reward = 0
        self.coverage_needs_update = True
        self.penalty  = 0 
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
        # Action execution
            if node_index < self.n_potential_nodes:
                rewards += self.deploy_node(node_index)
            else:
                rewards += self.keep_node()
                self.previous_actions.add(node_index)
        self.last_reward = rewards


        self.current_step += 1
        self.coverage_needs_update = False

        # 在环境初始化中添加
        # self.no_improvement_steps = 0

        # 在 step 函数中修改
        coverage_after = self.calculate_coverage_percentage()
        if coverage_after <= coverage_before and coverage_after < self.coverage_threshold:
            self.penalty = -120  
            # Adjust this penalty as needed
            # print(self.penalty)
            rewards += self.penalty

        if self.current_step >= self.max_steps:
            done = True
        elif coverage_after >= self.coverage_threshold:
            rewards += 10000
            done = True

        # Get next state
        next_state = self._get_flattened_state()

        return next_state, rewards, done

    def deploy_node(self, node_index):
        coverage_before = np.sum(self.coverage_grid)
        # print('coverage_before',coverage_before)
        if self.state["D"][node_index] == 1:
            # Node already deployed
            return self.calculate_reward()
        else:
            connected, donor_id = self.reconnect_node(node_index)
            if not connected:
                # Failed to connect to donor
                self.state["D"][node_index] = 0
                return self.calculate_reward()
            self.state["D"][node_index] = 1
            self.update_coverage_single_node(node_index)
            coverage_after = np.sum(self.coverage_grid)
            coverage_increase = coverage_after - coverage_before
            # print('coverage_increase',coverage_increase)
            # Optionally, reward based on coverage increase
            return self.calculate_reward()

    def keep_node(self):
        return self.calculate_reward()

    def reconnect_node(self, node_index):
        for d_id, donor_index in enumerate(self.donor_indices):
            best_target = None
            max_data_rate = 2.4
            for connection in self.connections_map.get(node_index, []):
                # 提取目标节点ID
                target = connection['node_id']
                
                # 确保 target 是整数
                if not isinstance(target, int):
                    try:
                        target = int(target)
                    except ValueError:
                        print(f"Invalid target value: {target}. Skipping.")
                        continue  # 跳过无效的 target
                
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
        # print('Data rate',self.state["R"])
        return True
    # def calculate_reward(self):
    #     coverage_before = self.previous_coverage
    #     coverage_after = self.calculate_coverage_percentage()
    #     coverage_gain = coverage_after - coverage_before
    #     reward = coverage_gain * 10  # Scale as needed
    #     self.previous_coverage = coverage_after
    #     # Penalize for excessive node deployment
    #     node_penalty = - (self.total_deployed_nodes() - self.optimal_node_count) * self.node_penalty_factor
    #     reward += node_penalty
    #     return reward

    def calculate_reward(self):
        alpha = 1
        beta = 0.00001
        uncovered_area_percent = 100 - self.calculate_coverage_percentage()
        uncovered_area_penalty = uncovered_area_percent * alpha
        deployment_penalty = beta * self.total_deployed_nodes()
        base_reward = -uncovered_area_penalty - deployment_penalty
        coverage = self.calculate_coverage_percentage()
        deployed_nodes = self.total_deployed_nodes()
        coverage_reward = base_reward
        coverage_increase = coverage - self.previous_coverage
        
        if coverage < 95:
            coverage_reward += (base_reward - (coverage / 95) * 10)  # Small linear reward below 95%
        elif 95 <= coverage < self.coverage_threshold:
            coverage_reward += coverage_increase * (1 + (coverage / 95) ** 2)

        # Penalty for excessive nodes
        
        if deployed_nodes > self.optimal_node_count:
            node_penalty = (deployed_nodes - self.optimal_node_count) * self.node_penalty_factor
        else:
            node_penalty = 0
        reward = coverage_reward+ node_penalty
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
        # Get positions of deployed nodes
        deployed_node_positions = [self.get_node_position(node_index) for node_index in deployed_nodes]

        # Minimum distance threshold to avoid deploying nodes too close to each other
        min_distance_threshold = 200  # Adjust this value as needed
        max_distance_threshold = 300  # Adjust this value as needed
        # Filter out nodes that are too close to deployed nodes
        valid_nodes = [node_index for node_index in (all_nodes - deployed_nodes)
                       if all(self.get_distance(self.get_node_position(node_index), pos) >= min_distance_threshold
                              for pos in deployed_node_positions)]

        # Ensure that 'do nothing' is always a valid action
        valid_actions = list(valid_nodes) + [self.n_potential_nodes]
        # print('valid-actions',len(valid_actions))
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
        """
        x, y = self.node_positions[node_index]
        return x, y

    def update_coverage_single_node(self, node_index):
        """
        Updates the coverage grid based on the coverage map of a deployed node.
        """
        if self.state["D"][node_index] == 1:
            for (y_idx, x_idx) in self.coverage_map.get(node_index, []):
                if 0 <= x_idx < self.map_size and 0 <= y_idx < self.map_size:
                    # Do not mark blocked areas as covered
                    if self.blocked_grid[y_idx, x_idx] == 0:
                        self.coverage_grid[y_idx, x_idx] = 1

    def calculate_coverage_percentage(self):
        """
        Calculates the percentage of the required area that is covered.

        The required coverage area is calculated based on the grid cells that need coverage,
        which are the cells not blocked by buildings (i.e., where self.blocked_grid == 0).
        """
        # Total number of cells that need to be covered (not blocked by buildings)
        required_coverage_area = np.sum(self.blocked_grid == 0)

        # Total number of cells that are currently covered
        total_covered = np.sum((self.coverage_grid == 1) & (self.blocked_grid == 0))

        # Calculate coverage percentage
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
        """Clean up the environment."""
        pass

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 800)
        self.fc2 = nn.Linear(800, 600)
        self.fc3 = nn.Linear(600, 400)
        self.fc4 = nn.Linear(400, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Agent:
    def __init__(self, state_dim, action_dim, learning_rate=0.0001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=140000, pretrained=False, model_path=None, freeze_layers=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=50000)
        self.total_steps = 0
        self.taken_actions = set()
        self.reward_mean = 0
        self.reward_std = 1
        self.alpha = 0.01
        self.pretrained = pretrained
        self.model_path = model_path
        self.freeze_layers = freeze_layers
        # self.episodes = episodes  # Total number of episodes

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Learning rate decay parameters
        # initial_lr = learning_rate
        # final_lr = initial_lr * 0.1  # Decay to 10% of initial_lr
        # lr_decay = (final_lr / initial_lr) ** (1 / self.episodes)

        # Initialize the learning rate scheduler
        # self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)

        if self.pretrained and self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path, self.freeze_layers)
            print(f"Loaded pretrained model from {self.model_path}")
        else:
            print("No pretrained model found, starting training from scratch.")
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Set different learning rates for different layers if freezing layers
        if self.freeze_layers:
            self.optimizer = optim.Adam([
                {'params': self.model.fc1.parameters(), 'lr': 0},  # Freeze layer
                {'params': self.model.ln1.parameters(), 'lr': 0},  # Freeze layer
                {'params': self.model.fc2.parameters(), 'lr': learning_rate},
                {'params': self.model.ln2.parameters(), 'lr': learning_rate},
                {'params': self.model.fc3.parameters(), 'lr': learning_rate},
                {'params': self.model.ln3.parameters(), 'lr': learning_rate},
                {'params': self.model.fc4.parameters(), 'lr': learning_rate},
            ], lr=learning_rate)
            print("Set different learning rates for different layers.")
        else:
            # 在Agent类中
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

            # self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5000, verbose=True)
        self.taken_actions = set()


    def reset_actions(self):
        self.taken_actions.clear()
        
    def load_model(self, path, freeze_layers=False):
        """
        Loads the model's state dictionary from the specified path and freezes layers if needed.
        """
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"Model loaded from {path}.")

        if freeze_layers:
            # Freeze the parameters of the first layer (fc1 and ln1)
            for param in self.model.fc1.parameters():
                param.requires_grad = False
            for param in self.model.ln1.parameters():
                param.requires_grad = False
            print("Frozen parameters of fc1 and ln1 layers.")

        # Set the optimizer to optimize only parameters that require gradients
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        #self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5000, verbose=True)

    def reset_actions(self):
        """Reset taken actions and other relevant parameters if needed."""
        self.taken_actions = set()

    def check_frozen_layers(self):
        """
        Checks which layers' parameters are frozen.
        """
        frozen = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                frozen.append(name)
        print(f"Frozen layers: {frozen}")

    def select_action(self, state, valid_actions):
        """
        Selects an action using epsilon-greedy strategy, considering only valid actions.
        """
        state = self.normalize_state(state)
        do_nothing_action = self.action_dim -1 # Assuming 'do nothing' is the last action index

        # Ensure 'do nothing' action is always in valid_actions
        if do_nothing_action not in valid_actions:
            valid_actions.append(do_nothing_action)

        # Exclude taken actions, but always include 'do nothing' action
        filtered_valid_actions = [
            action for action in valid_actions
            if (action not in self.taken_actions) or (action == do_nothing_action)
        ]

        if not filtered_valid_actions:
            # No valid actions left, select 'do nothing'
            action_index = do_nothing_action
        else:
            eps_threshold = self.get_epsilon()
            if random.random() < eps_threshold:
                action_index = random.choice(filtered_valid_actions)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = self.model(state_tensor).squeeze().cpu().numpy()
                # Select the action with the highest Q-value among valid actions
                valid_q_values = q_values[filtered_valid_actions]
                max_q_index = np.argmax(valid_q_values)
                action_index = filtered_valid_actions[max_q_index]

        # Only add actions other than 'do nothing' to taken_actions
        if action_index != do_nothing_action:
            self.taken_actions.add(action_index)
        return action_index


    # def select_action(self, state, valid_actions):
    #     """
    #     Selects an action using epsilon-greedy strategy, considering only valid actions.
    #     """
    #     state = self.normalize_state(state)
    #     valid_actions = [action for action in valid_actions if action not in self.taken_actions]
    #     eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
    #         -1. * self.total_steps / self.epsilon_decay)
    #     if random.random() < eps_threshold:
    #         # Ensure there's always at least one valid action
    #         action_index = random.choice(valid_actions)
    #     else:
    #         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    #         with torch.no_grad():
    #             q_values = self.model(state_tensor)
    #         q_values = q_values.squeeze().cpu().numpy()
    #         # Select the action with the highest Q-value among valid actions
    #         # Mask invalid actions
    #         q_values_masked = np.full_like(q_values, -np.inf)
    #         q_values_masked[valid_actions] = q_values[valid_actions]
    #         action_index = np.argmax(q_values_masked)
    #         self.taken_actions.add(action_index)
    #     return action_index
        #     valid_q_values = q_values[valid_actions]
        #     max_q_index = np.argmax(valid_q_values)
        #     action_index = valid_actions[max_q_index]

        # return action_index

    def save_model(self, path):
        """
        Saves the model's state dictionary to the specified path.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def normalize_reward(self, reward):

        # Update running mean and std
        self.reward_mean = (1 - self.alpha) * self.reward_mean + self.alpha * reward
        self.reward_std = (1 - self.alpha) * self.reward_std + self.alpha * (reward - self.reward_mean) ** 2
        reward_normalized = (reward - self.reward_mean) / (math.sqrt(self.reward_std) + 1e-5)
        return reward_normalized

    def store_transition(self, state, action, reward, next_state, done):
        # Normalize reward
        reward_normalized = self.normalize_reward(reward)
        self.memory.append((state, action, reward_normalized , next_state, done))


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
        loss = F.mse_loss(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        #self.scheduler.step(loss)

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

def train(env, agent, episodes, batch_size, target_update, save_model_interval, pretrained_model_path=None):
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

        if pretrained_model_path and (episode + 1) == pretrained_model_path.get('episode', -1):
            agent.load_model(pretrained_model_path['path'],
                             freeze_layers=pretrained_model_path.get('freeze_layers', True))
            print(f"Transferred learning from episode {pretrained_model_path['episode']}")

        # Clear experience replay buffer if using transfer learning
        if episode == 0 and agent.pretrained:
            agent.memory.clear()
            print("Cleared experience replay buffer to adapt to new environment.")

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

        # Periodically save the model
        if (episode + 1) % save_model_interval == 0 and (episode + 1) != 0:
            model_save_path = f'model_episode_{episode + 1}.pth'
            agent.save_model(model_save_path)
            print(f'Model saved at episode {episode + 1}')

        # Log each episode's results
        print(f'End of Episode {episode + 1}:')
        print(f'Total Reward: {total_reward}')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Total Deployed Nodes: {deployed_nodes}')
        print(f'Coverage: {coverage_percentage:.2f}%')
        print(f'Epsilon: {epsilon:.4f}')
        print(f'Learning Rate: {current_lr:.6f}\n')

    # Save the final model
    agent.save_model('model_final_DQN.pth')
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
                [data['Episode'][i], data['Total Reward'][i], data['Avg Loss'][i],
                 data['Deployed Nodes'][i], data['Coverage'][i]]
            )

    return episode_rewards, episode_losses, episode_coverages  # Returns coverage

def main():
    # Parameters
    max_steps = 250
    number_of_donors = 5
    episodes = 6000  # Reduced for faster training during testing
    batch_size = 256
    target_update = 64
    save_model_interval = 20000

    # Transfer Learning Parameters
    use_transfer_learning = False  # Set to True to use transfer learning
    pretrained_model_info = {
        'path': 'model_final_DQN.pth',  # Path to the pre-trained model
        'freeze_layers': False  # Whether to freeze layers
    } if use_transfer_learning else None

    freeze_layers = False

    # Initialize environment
    env = NetworkDeploymentEnv(
        map_size=1000,
        coverage_threshold=100,  # Adjusted to 100%
        optimal_node_count=26,
        node_penalty_factor=5,
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
        model_path=pretrained_model_info['path'] if pretrained_model_info else None,
        freeze_layers=freeze_layers
    )
    agent.check_frozen_layers()
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

def plot_and_save_results(rewards, losses, coverages, timestamp):
    """
    Plots and saves the rewards, losses, and coverage percentages over episodes,
    including a 300-episode moving average for each.
    """
    window_size = 300  # Moving window size

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # Plot total rewards per episode
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Total Reward per Episode')
    # Compute and plot moving average
    if len(rewards) >= window_size:
        rewards_moving_avg = moving_average(rewards, window_size)
        plt.plot(range(window_size - 1, len(rewards)), rewards_moving_avg, label=f'{window_size}-Episode Moving Average')
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'total_rewards_{timestamp}.png')
    plt.close()

    # Plot average losses per episode
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Average Loss per Episode')
    # Compute and plot moving average
    if len(losses) >= window_size:
        losses_moving_avg = moving_average(losses, window_size)
        plt.plot(range(window_size - 1, len(losses)), losses_moving_avg, label=f'{window_size}-Episode Moving Average')
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'average_losses_{timestamp}.png')
    plt.close()

    # Plot coverage percentages per episode
    plt.figure(figsize=(10, 6))
    plt.plot(coverages, label='Coverage Percentage per Episode')
    # Compute and plot moving average
    if len(coverages) >= window_size:
        coverages_moving_avg = moving_average(coverages, window_size)
        plt.plot(range(window_size - 1, len(coverages)), coverages_moving_avg, label=f'{window_size}-Episode Moving Average')
    plt.title('Coverage Percentage per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Coverage (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'coverage_percentages_{timestamp}.png')
    plt.close()

if __name__ == "__main__":
    main()