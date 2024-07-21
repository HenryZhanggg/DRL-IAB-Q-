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
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=70000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
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


    def select_action(self, state, valid_actions):
        # Filter valid actions to remove those already taken
        state = self.normalize_state(state)
        valid_actions = [action for action in valid_actions if action not in self.taken_actions]

        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.total_steps / self.epsilon_decay)
        if random.random() < eps_threshold or not valid_actions:
            # Ensure there's always at least one valid action
            action_index = random.choice(list(set(range(self.action_dim)) - self.taken_actions))
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state)
            q_values = q_values.squeeze().cpu().detach().numpy()
            valid_q_values = {action: q_values[action] for action in valid_actions}
            action_index = max(valid_q_values, key=valid_q_values.get)

        self.taken_actions.add(action_index)  # Record action as taken
        return action_index

    # def select_action(self, state, valid_actions):
    #     eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.total_steps / self.epsilon_decay)
    #     if random.random() < eps_threshold:
    #         action_index = random.choice(valid_actions)
    #     else:
    #         state = torch.FloatTensor(state).unsqueeze(0).to(device)
    #         q_values = self.model(state)
    #         q_values = q_values.squeeze().cpu().detach().numpy()
    #         valid_q_values = {action: q_values[action] for action in valid_actions}
    #         action_index = max(valid_q_values, key=valid_q_values.get)
    #     return action_index

    def reset_actions(self):
        self.taken_actions.clear()

    def store_transition(self, state, action, reward, next_state, done):
        # actual_episode_rewards.append(reward)  # Store actual rewards for plotting
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

        # Double DQN update
        next_actions_online = self.model(next_states).max(1)[1].unsqueeze(1)
        next_q_values_target = self.target_model(next_states).gather(1, next_actions_online).squeeze()

        expected_q_values = rewards + (self.gamma * next_q_values_target * (1 - dones))
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
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.total_steps / self.epsilon_decay)
    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    

def train(env, agent, episodes, batch_size=512, target_update=64, scale=300,save_model_interval=30000):
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
        # print("episode: ", episode)
        while not done:
            start_step_time = time.time()
            valid_actions = env.get_valid_actions()
            # print(f"Valid actions: {valid_actions}")
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
            # print(f'Step {step_count} execution time: {end_step_time - start_step_time} seconds')

            if step_count >= max_steps:
                done = True
            elif env.calculate_coverage_percentage() >= env.coverage_threshold:
                done = True

        end_episode_time = time.time()
        # print(f'Episode {episode} execution time: {end_episode_time - start_episode_time} seconds')

        avg_loss = total_loss / step_count
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
        # Save model periodically
        if episode % save_model_interval == 0:
            torch.save(agent.model.state_dict(), f'model_episode_{episode}.pth')
            print(f'Model saved at episode {episode}')
        # Print results after each episode
        print(f'End of Episode {episode}:')
        print(f'Total Reward: {total_reward}')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Total Deployed Nodes: {deployed_nodes}')
        print(f'Coverage: {coverage_percentage:.2f}')
        print(f'Epsilon: {epsilon:.4f}')
        print(f'Learning Rate: {current_lr:.6f}\n')

        # Save final model
    torch.save(agent.model.state_dict(), 'model_finalDDQN.pth')
    print('Final model saved.')
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

    # Plot Reward
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Reward per Episode')
    plt.plot(episodes_scale, avg_rewards_per_scale_episodes, label='Avg Reward per 100 Episodes', color='red', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.ylim([min(episode_rewards), max(episode_rewards)])  # Set a larger Y-axis scale
    plt.title('Episode vs Reward')
    plt.legend()
    filename = f'Episode_vs_Reward_{timestamp}.png'
    plt.savefig(filename)
    plt.close()

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(episode_losses, label='Loss per Episode')
    plt.plot(episodes_scale, avg_losses_per_scale_episodes, label='Avg Loss per 100 Episodes', color='blue', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.ylim([min(episode_losses), max(episode_losses)])  # Set a larger Y-axis scale
    plt.title('Episode vs Loss')
    plt.legend()
    filename = f'Episode_vs_Loss_{timestamp}.png'
    plt.savefig(filename)
    plt.close()

    # Plot Deployed Nodes and Coverage Percentage
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(data["Episode"], data["Deployed Nodes"], label='Deployed Nodes per Episode', color='green', marker='o', linestyle='-', linewidth=1, markersize=4)
    plt.plot(episodes_scale, avg_numofnodes_per_scale_episodes, label='Avg nodes per 100 Episodes', color='red', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Number of Deployed Nodes')
    plt.ylim([min(data["Deployed Nodes"]), max(data["Deployed Nodes"])])  # Set a larger Y-axis scale
    plt.title('Episode vs Number of Deployed Nodes')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(data["Episode"], data["Coverage"], label='Coverage Percentage per Episode', color='purple', marker='o', linestyle='-', linewidth=1, markersize=4)
    plt.plot(episodes_scale, avg_coverage_per_scale_episodes, label='Avg Coverage per 100 Episodes', color='red', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Coverage Percentage')
    plt.ylim([95, 100])  # Set a larger Y-axis scale focusing on 95-100
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
                [data['Episode'][i], data['Total Reward'][i], data['Avg Loss'][i], data['Deployed Nodes'][i],
                 data['Coverage'][i]])

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
        self.map_size = 1000
        self.grid_size = 1000
        self.node_spacing = 50
        self.n_potential_nodes_per_row = int(self.grid_size // self.node_spacing)
        self.n_potential_nodes = int(self.grid_size // self.node_spacing) ** 2
        self.action_space = self.n_potential_nodes + 1
        self.coverage_grid = np.zeros((self.map_size, self.map_size), dtype=np.int8)
        self.observation_space = spaces.Dict({
            "D": spaces.MultiBinary(self.n_potential_nodes),
            "R": spaces.Box(low=0, high=50, shape=(self.n_potential_nodes,)),
            "N": spaces.Box(low=0, high=10, shape=(self.n_potential_nodes,))
        })
        self.overhead = 1.2
        self.node_data_rate = 2
        self.donor_data_rate = 50  # 15 Gbps
        self.coverage_radius = 200
        self.backhaul_radius = 300
        self.narrow = 1
        self.numberofdonor = numberofdonor
        self.current_step = 0
        self.max_steps = max_steps
        self.previous_actions = set()
        self.last_reward = None
        self.coverage_needs_update = True
        self.donor_adjacency_matrices = np.zeros((self.numberofdonor, self.n_potential_nodes, self.n_potential_nodes))
        self.donor_indices = self.get_fixed_donor_positions()
        self.previous_coverage = 0.0  # To store the previous coverage percentage
        self.optimal_node_count = 22  # Set this based on your specific problem
        self.node_penalty_factor = 100  # Adjust this to balance coverage vs. efficiency
        self.coverage_threshold = 100  # The target coverage percentage
        # self.base_reward = -100  # Base negative reward to encourage efficiency

        self.reset()

    def get_fixed_donor_positions(self):
        # Define fixed donor positions here (example: first few nodes in the grid)
        fixed_positions = [0, 50, 90, 150, 210]
        # fixed_positions = [0, 10, 20, 30, 40]  # These are indices, modify as needed
        # fixed_positions = [359, 369, 379, 389, 399]
        return fixed_positions

    def load_precomputed_data(self):
        try:
            with open('coverage_map.pkl1002', 'rb') as f:
                self.coverage_map = pickle.load(f)
            with open('connections_map.pkl1002', 'rb') as f:
                self.connections_map = pickle.load(f)
        except FileNotFoundError:
            print("Failed to load the precomputed data files. Please check the files' existence and paths.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def reset(self):
        self.state = {
            "D": np.zeros(self.n_potential_nodes),
            "R": np.zeros(self.n_potential_nodes),
            "N": np.zeros((self.n_potential_nodes))}
        self.state["N"] = np.zeros((self.n_potential_nodes))
        self.state["R"] = np.zeros(self.n_potential_nodes)
        self.coverage_grid.fill(0)
        self.donor_adjacency_matrices.fill(0)
        # self.donor_indices = np.random.choice(range(self.n_potential_nodes), self.numberofdonor, replace=False)
        for idx in self.donor_indices:
            self.state["D"][idx] = 1  # Mark as deployed
            self.state["R"][idx] = 50
            self.update_coverage_single_node(idx)
        self.current_step = 0
        self.previous_actions = set()
        self.last_reward = 0
        self.coverage_needs_update = True
        # print('initial coverage', np.sum(self.coverage_grid))
        return self._get_flattened_state()

    def update_coverage_single_node(self, node_index):
        if self.state["D"][node_index] == 1:
            for (x, y) in self.coverage_map[node_index]:
                self.coverage_grid[x, y] = 1
        # print('added coverage', np.sum(self.coverage_grid))

    def step(self, action_index):
        coverage_before = self.calculate_coverage_percentage()
        valid_actions = self.get_valid_actions()
        if action_index not in valid_actions:
            action_index = valid_actions[np.random.randint(len(valid_actions))]
        rewards = 0
        done = False
        node_index = action_index
        # print(f"Action: deploy node at position {node_index}")
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

        if coverage_after <= coverage_before and coverage_after < self.coverage_threshold:
            penalty = -5000  # You can adjust this value based on your needs
            rewards += penalty
            # print(f"Applied penalty: {penalty} for not increasing coverage. Before: {coverage_before:.2f}%, After: {coverage_after:.2f}%")

        if self.current_step >= max_steps:
            self.current_step = 0
            self.coverage_needs_update = True
            done = True
        elif self.calculate_coverage_percentage() >= self.coverage_threshold:
            self.current_step = 0
            self.coverage_needs_update = True
            done = True

        return self._get_flattened_state(), rewards, done

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
                self.update_coverage()
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

    def calculate_uncovered_areas(self):
        uncovered_positions = []
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.coverage_grid[i, j] == 0:
                    uncovered_positions.append((i, j))
        return uncovered_positions

    def get_nodes_near_uncovered_areas(self, uncovered_positions, radius=200):
        valid_nodes = set()
        for x, y in uncovered_positions:
            for node_index in range(self.n_potential_nodes):
                node_x, node_y = self.get_node_position(node_index)
                distance = np.sqrt((node_x - x) ** 2 + (node_y - y) ** 2)
                if distance <= radius:
                    valid_nodes.add(node_index)
        return list(valid_nodes)

    # def get_valid_actions(self):
    #     uncovered_positions = self.calculate_uncovered_areas()
    #     valid_nodes = self.get_nodes_near_uncovered_areas(uncovered_positions)
    #     valid_actions = valid_nodes + [self.n_potential_nodes]  # Include the 'do nothing' action
    #     return valid_actions
    def get_valid_actions(self):
        valid_nodes = set(range(self.n_potential_nodes))
        deployed_nodes = set(np.where(self.state["D"] == 1)[0])
        valid_nodes = valid_nodes - deployed_nodes
        # print('valid_nodes',len(valid_nodes))
        # Get the positions of deployed nodes
        deployed_node_positions = [self.get_node_position(node_index) for node_index in deployed_nodes]
        # print('deployed_node_positions',deployed_node_positions)
        # Calculate the current coverage percentage
        coverage_percentage = self.calculate_coverage_percentage()
        max_distance_threshold = 300
        # Adjust the minimum distance threshold based on the coverage percentage
        min_distance_threshold = 250  # Initial threshold
        if coverage_percentage >= 90:
            min_distance_threshold = 200  # Relax the threshold when coverage is high
        elif coverage_percentage >= 60:
            min_distance_threshold = 200  # Intermediate threshold

        # for node_index in valid_nodes:
        #     print('self.get_node_position(node_index)',self.get_node_position(node_index))
        # Filter out nodes that are too close to previously deployed nodes
        valid_nodes = [node_index for node_index in valid_nodes
                       if all(min_distance_threshold <= self.get_distance(self.get_node_position(node_index), pos)
                              for pos in deployed_node_positions)]

        valid_actions = list(valid_nodes) + [self.n_potential_nodes]  # Include the 'do nothing' action
        # print('valid_actions',valid_actions)
        return valid_actions

    # def get_valid_actions(self):
    #     valid_nodes = set(range(self.n_potential_nodes))

    #     deployed_nodes = set(np.where(self.state["D"] == 1)[0])
    #     valid_nodes = valid_nodes - deployed_nodes
    #     print('valid_nodes0-1',valid_nodes)
    #     # Get the positions of deployed nodes
    #     deployed_node_positions = [self.get_node_position(node_index) for node_index in deployed_nodes]

    #     # Calculate the current coverage percentage
    #     coverage_percentage = self.calculate_coverage_percentage()

    #     # Adjust the minimum distance threshold based on the coverage percentage
    #     min_distance_threshold = 100  # Initial threshold
    #     max_distance_threshold = 300  # Maximum threshold

    #     # Filter out nodes that are too close to previously deployed nodes
    #     valid_nodes = [node_index for node_index in valid_nodes
    #                    if all(min_distance_threshold <= self.get_distance(self.get_node_position(node_index), pos) <= max_distance_threshold
    #                           for pos in deployed_node_positions)]

    #     valid_actions = list(valid_nodes) + [self.n_potential_nodes]  # Include the 'do nothing' action

    #     return valid_actions

    def _get_flattened_state(self):
        D_flat = self.state["D"].flatten()
        R_flat = self.state["R"].flatten()
        N_flat = self.state["N"].flatten()
        flattened_state = np.concatenate([D_flat, R_flat, N_flat])
        return flattened_state

    def get_distance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def render(self, mode='human'):
        pass

    def total_deployed_nodes(self):
        total_deployed_nodes = np.sum(self.state["D"])
        # print("tol_deployed_node: ", total_deployed_nodes)
        return total_deployed_nodes

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
        if coverage < 95:
            coverage_reward += (base_reward - (coverage / 95) * 10)  # Small linear reward below 95%
        elif 95 <= coverage < self.coverage_threshold:
            # Exponential reward between 95% and 98%
            coverage_reward += (base_reward + math.exp(coverage - 95))

        # Penalty for excessive nodes
        if deployed_nodes > self.optimal_node_count:
            node_penalty = (deployed_nodes - self.optimal_node_count) * self.node_penalty_factor
        else:
            node_penalty = 0
        reward = coverage_reward - node_penalty
        # print('reward',reward)
        return reward


    def update_coverage(self):
        for node_index, is_deployed in enumerate(self.state['D']):
            if self.state["D"][node_index] == 1:
                for (x, y) in self.coverage_map[node_index]:
                    self.coverage_grid[x, y] = 1
                # print('coverage increase', np.sum(self.coverage_grid))

    def calculate_coverage_percentage(self):
        total_covered = np.sum(self.coverage_grid)
        total_area = self.grid_size * self.grid_size
        coverage_percentage = (total_covered / total_area) * 100
        # print('coverage_percentage', coverage_percentage)
        return coverage_percentage

    def get_node_position(self, node_index):
        row = (node_index) // (self.n_potential_nodes_per_row)
        col = (node_index) % (self.n_potential_nodes_per_row)
        x = int(col * self.node_spacing * self.narrow)
        y = int(row * self.node_spacing * self.narrow)
        return x, y

max_steps = 50
scale = 100
numberofdonor = 5
env = NetworkDeploymentEnv()
state_dim = len(env.reset())
action_dim = env.action_space
agent = Agent(state_dim, action_dim)
rewards = train(env, agent, episodes=30000)