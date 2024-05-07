#1000*1000 with 4 meters grid
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
# device = torch.device("cpu")

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Enhanced MLP with additional layers and normalization
        self.fc1 = nn.Linear(n_observations, 1024)  # Increased from 512 to 1024
        self.ln1 = nn.LayerNorm(1024)  # Adjusted LayerNorm to new size
        self.fc2 = nn.Linear(1024, 1024)  # Added another 1024 layer for depth
        self.ln2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, 512)  # New intermediary layer
        self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, 256)   # Maintaining this layer
        self.ln4 = nn.LayerNorm(256)
        self.fc5 = nn.Linear(256, n_actions)  # Final layer outputting actions

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        x = F.relu(self.ln4(self.fc4(x)))
        return self.fc5(x)

class Agent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=250000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DQN(state_dim, action_dim).to(device)

        self.target_model = copy.deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon_start = epsilon_start  # Renamed from self.epsilon to self.epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay  # Use this for the decay calculation
        self.memory = deque(maxlen=100000)
        self.total_steps = 0

    def select_action(self, state):
        print('self.total_steps1',self.total_steps)
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.total_steps / self.epsilon_decay)
        if random.random() < eps_threshold:
            action_index = np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state)
            action_index = q_values.argmax().item()
        return action_index

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state,done))
        #print('self.memory.reward',reward)

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        #print('batch_siz:rewards',rewards)
        states = torch.tensor(np.array(states, dtype=np.float32), device=device, dtype=torch.float32)
        #states = torch.FloatTensor(np.array(states)).to(device)  # Move to device
        actions = torch.LongTensor(actions).to(device)  # Move to device
        rewards = torch.FloatTensor(rewards).to(device)  # Move to device
        next_states = torch.FloatTensor(np.array(next_states)).to(device)  # Move to device
        dones = torch.FloatTensor(dones).to(device)
        # Get current Q values for chosen actions

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        # Compute the expected Q values
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        #expected_q_values = rewards + (self.gamma * next_q_values)
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), expected_q_values)
        #print("LOSS: ", loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

def train(env, agent, episodes, batch_size=64, target_update=64, scale=100):
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
        start_episode_time = time.time()  # 开始测量一个episode的时间
        state = env.reset()
        total_reward = 0
        total_loss = 0
        done = False
        step_count = 0
        print("episode: ", episode)
        while not done:
            start_step_time = time.time()  # 开始测量一个step的时间
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.experience_replay(batch_size)
            #print('step', step_count)
            state = next_state
            #print('state after flattern',state)
            total_reward += reward
            total_loss += loss if loss is not None else 0
            #print("==> tol loss", total_loss)
            step_count += 1
            agent.total_steps += 1

            # Collecting step details
            step_details = {
                "Episode": episode,
                "Step": step_count,
                "Action": action,
                "Reward": reward,
                "Coverage": env.calculate_coverage_percentage()
            }
            all_steps_details.append(step_details)
            print('agent.total_steps',agent.total_steps)
            print('step_count', step_count)
            if agent.total_steps % target_update == 0:
                agent.update_target_network()
            end_step_time = time.time()  # 结束测量一个step的时间
            print(f'Step {step_count} execution time: {end_step_time - start_step_time} seconds')

            if step_count >= max_steps:
                print(f"reach maximum steps {step_count}")
                # step_count = 0
                done = True

            elif env.calculate_coverage_percentage() >= 100:
                print(f"100% Coverage achieved at episode {episode} step {step_count}")
                # step_count = 0
                done = True
        end_episode_time = time.time()  # 结束测量一个episode的时间
        print(f'Episode {episode} execution time: {end_episode_time - start_episode_time} seconds')

        print('total_loss',total_loss)
        avg_loss = total_loss / step_count #if step_count else 0
        print('avg_loss', avg_loss)
        #print("avg_loss: ", avg_loss,step_count)
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        deployed_nodes = env.total_deployed_nodes()
        print('total deployed node in this eposide',env.total_deployed_nodes())
        coverage_percentage = env.calculate_coverage_percentage()

        data["Episode"].append(episode)
        data["Total Reward"].append(total_reward)
        data["Avg Loss"].append(avg_loss)
        print("Avg loss saved: ", data["Avg Loss"])
        data["Deployed Nodes"].append(deployed_nodes)
        data["Coverage"].append(coverage_percentage)

    # Calculating averages per scale of episodes
    avg_rewards_per_scale_episodes = [np.mean(episode_rewards[i:i+scale]) for i in range(0, len(episode_rewards), scale)]
    avg_losses_per_scale_episodes = [np.mean(episode_losses[i:i+scale]) for i in range(0, len(episode_losses), scale)]
    avg_numofnodes_per_scale_episodes = [np.mean(data["Deployed Nodes"][i:i + scale]) for i in range(0, len(data["Deployed Nodes"]), scale)]
    avg_coverage_per_scale_episodes = [np.mean(data["Coverage"][i:i + scale]) for i in range(0, len(data["Coverage"]), scale)]
    episodes_scale = list(range(0, len(episode_rewards), scale))

    # Writing step details to CSV
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
    plt.plot(episodes_scale, avg_rewards_per_scale_episodes, label='Avg Reward per 100 Episodes', color='red', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Episode vs Reward')
    # Generate a timestamp or unique identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'Episode vs Reward_Time_{timestamp}.png'
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory
    plt.legend()

    plt.plot(episode_losses, label='Loss per Episode')
    plt.plot(episodes_scale, avg_losses_per_scale_episodes, label='Avg Loss per 100 Episodes', color='blue', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Episode vs Loss')
    #plt.savefig('Episode vs Loss for removing all.png')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'Episode vs Loss for removing all_Time_{timestamp}.png'
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(data["Episode"], data["Deployed Nodes"], label='Deployed Nodes per Episode', color='green', marker='o',
             linestyle='-', linewidth=1, markersize=4)
    plt.xlabel('Episodes')
    plt.ylabel('Number of Deployed Nodes')
    plt.plot(episodes_scale, avg_numofnodes_per_scale_episodes, label='Avg nodes per 100 Episodes', color='red',
             linewidth=2)
    plt.title('Episode vs Number of Deployed Nodes removing all')
    plt.legend()

    # Plot Coverage Percentage vs. Episode
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'Episode vs Coverage Percentage removing all_Time_{timestamp}.png'
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory
    #plt.show()
    # After the training loop, write data to a CSV file

    filename = f'training_data removing all {timestamp}.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['Episode', 'Total Reward', 'Avg Loss', 'Deployed Nodes', 'Coverage'])
        # Write the data
        for i in range(len(data['Episode'])):
            writer.writerow([data['Episode'][i], data['Total Reward'][i], data['Avg Loss'][i], data['Deployed Nodes'][i], data['Coverage'][i]])

    return episode_rewards, episode_losses

def print_initial_R_N(state, n_potential_nodes):
    R_start = n_potential_nodes
    N_start = 2 * n_potential_nodes
    #print("Initial R:", state[R_start:R_start + n_potential_nodes])
    #print("Initial N row 0:", state[N_start:N_start + n_potential_nodes])  # Example: first row


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
            "R": spaces.Box(low=0, high=20, shape=(self.n_potential_nodes,)),
            "N": spaces.MultiBinary((self.n_potential_nodes, self.n_potential_nodes))
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
        self.dor_coverage_cache = None  # Initialize the donor coverage cache
        self.coverage_needs_update = True
        self.reset()

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
            "N": np.zeros((self.n_potential_nodes, self.n_potential_nodes))}
        self.state["N"] = np.zeros((self.n_potential_nodes, self.n_potential_nodes))
        self.state["R"] = np.zeros(self.n_potential_nodes)
        self.coverage_grid.fill(0)
        # Randomly choose 10 donor locations
        self.permanent_donors = np.random.choice(range(self.n_potential_nodes), numberofdonor, replace=False)
        for idx in self.permanent_donors:
            self.state["D"][idx] = 1  # Mark as deployed
            self.state["R"][idx] = 15
            self.update_coverage_single_node(idx)
        self.current_step = 0
        self.previous_actions = set()
        self.last_reward = 0
        self.coverage_needs_update = True

        print('initial coverage', np.sum(self.coverage_grid))
        return self._get_flattened_state()

    def update_coverage_single_node(self, node_index):
        # This function updates the coverage grid for a single node
        if self.state["D"][node_index] == 1:
            for (x, y) in self.coverage_map[node_index]:
                self.coverage_grid[x, y] = 1
        print('added coverage', np.sum(self.coverage_grid))

    def step(self, action_index):
        rewards = 0
        done = False
        node_index = action_index
        print(f"Action: deploy node at position {node_index}")
        #print('Before action state', self.state["D"])
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
        if self.current_step >= max_steps:
            #print(f"reach maximum steps")
            self.current_step = 0
            self.coverage_needs_update = True
            done = True
        elif env.calculate_coverage_percentage() >= 100:
            #print(f"100% Coverage achieved at step")
            self.current_step = 0
            self.coverage_needs_update = True
            done = True

        return self._get_flattened_state(), rewards, done

    def deploy_node(self, node_index):
        node_x, node_y = self.get_node_position(node_index)  # Get node position
        print(f"Attempting to deploy node at position: ({node_x}, {node_y})")

        # Print coverage before deployment
        coverage_before = np.sum(self.coverage_grid)
        print(f"Coverage before deployment: {coverage_before} grids")
        if self.state["D"][node_index] == 1:
            self.state["D"][node_index] = 1
            self.update_coverage()
            print(f"Node {node_index} at position ({node_x}, {node_y}) is already deployed.")
            return self.calculate_reward()
        else:
            connected = self.reconnect_node(node_index)
            if not connected:
                self.state["D"][node_index] = 0
                self.update_coverage()
                print(f"Failed to connect node at index {node_index}.")
                return self.calculate_reward()
            self.state["D"][node_index] = 1
            self.update_coverage_single_node(node_index)
            coverage_after = np.sum(self.coverage_grid)
            coverage_increase = coverage_after - coverage_before
            print(f"Node deployed at ({node_x}, {node_y}). New coverage: {coverage_increase} additional grids.")
            return self.calculate_reward()  # Return rewards considering new coverage

    def keep_node(self):
        return self.calculate_reward()

    def reconnect_node(self, node_index):
        if node_index not in self.connections_map:
            print(f"No connections data for node {node_index}.")
            return False
        best_target = None
        max_data_rate = 2.4
        for target in self.connections_map[node_index]:
            if self.state['D'][target] == 1 and self.state['R'][target] > max_data_rate:
                max_data_rate = self.state['R'][target]
                best_target = target
        #print('best targets data rate and node', self.state['R'][best_target], self.state['D'][best_target])
        if best_target is not None:
            self.state["N"][best_target][node_index] = 1  # 根据网络模型可能需要调整为单向连接
            self.state["R"][best_target] -= self.node_data_rate * self.overhead
            # 遍历所有节点，找到服务best target的节点，并减少它们的data rate
            for j in range(self.n_potential_nodes):
                if self.state["N"][best_target][j] == 1:  # 如果best_target与i节点连接
                    self.state["R"][j] -= self.node_data_rate * self.overhead  # 调整data rate
            #print(f"Reconnected node {node_index} to node {best_target} with data rate {self.state['R'][best_target]}")
            self.state["R"][node_index] = self.state["R"][best_target]
            return True
        return False

    def _get_flattened_state(self):
        # Flatten the state components into a single array
        D_flat = self.state["D"].flatten()
        R_flat = self.state["R"].flatten()
        N_flat = self.state["N"].flatten()
        flattened_state = np.concatenate([D_flat, R_flat, N_flat])
        #print('flattened_state',flattened_state)
        return flattened_state

    def render(self, mode='human'):
        pass

    def total_deployed_nodes(self):
        total_deployed_nodes = np.sum(self.state["D"])
        print("tol_deployed_node: ", total_deployed_nodes)
        return total_deployed_nodes

    def calculate_reward(self):
        alpha = 100 # Penalty for uncovered area
        #beta = 0.5  # Penalty for each deployed node
        beta = 0.1  # Penalty for each deployed node
        #gamma = 100  # Reward multiplier for coverage

        uncovered_area_percent = 100 - env.calculate_coverage_percentage()
        # Calculate penalties and rewards
        uncovered_area_penalty = uncovered_area_percent * alpha
        #print('uncovered_area_penalty:', uncovered_area_penalty)
        #print('deployed nodes after this action', self.total_deployed_nodes())
        deployment_penalty = beta * self.total_deployed_nodes()

        #coverage_reward = gamma * (covered_area / total_area)  # Reward based on the percentage of area covered

        # Final reward is thOld N row e coverage reward minus penalties
        #reward = (100*env.calculate_coverage_percentage())/self.total_deployed_nodes()
        reward = -uncovered_area_penalty - deployment_penalty
        #print('reward:',reward)
        return reward

    def update_coverage(self):
        for node_index, is_deployed in enumerate(self.state['D']):
            if self.state["D"][node_index] == 1:
                for (x, y) in self.coverage_map[node_index]:
                    self.coverage_grid[x, y] = 1
                print('coverage increase',np.sum(self.coverage_grid))


    def calculate_coverage_percentage(self):
        total_covered = np.sum(self.coverage_grid)
        total_area = self.grid_size * self.grid_size
        coverage_percentage = (total_covered / total_area) * 100
        print('coverage_percentage', coverage_percentage)
        return coverage_percentage

    def get_node_position(self, node_index):
        row = (node_index) // (self.n_potential_nodes_per_row)
        col = (node_index) % (self.n_potential_nodes_per_row)
        x = int(col * self.node_spacing * self.narrow)
        y = int(row * self.node_spacing * self.narrow)
        return x, y

    def calculate_distance(self, node_index1, node_index2):
        x1, y1 = self.get_node_position(node_index1)
        x2, y2 = self.get_node_position(node_index2)
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

max_steps = 100
scale = 100
numberofdonor = 5
env = NetworkDeploymentEnv()
state_dim = len(env.reset())
action_dim = env.action_space
agent = Agent(state_dim, action_dim)
rewards = train(env, agent, episodes=20000)
