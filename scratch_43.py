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

import matplotlib.pyplot as plt
import numpy as np
import copy
import csv
import datetime

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

class Agent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_step=0.995, epsilon_decay_episode=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_step = epsilon_decay_step
        self.memory = deque(maxlen=200000)
        self.total_steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            action_index = np.random.randint(self.action_dim)
            return action_index
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Move state to the device
            q_values = self.model(state)
            action_index = q_values.argmax().item()
        self.epsilon = max(self.epsilon_end,
                               self.epsilon * self.epsilon_decay_step)  # Exponential decay per step

        return action_index

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #print('self.memory.reward',reward)

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        #print('batch_siz:rewards',rewards)
        states = torch.FloatTensor(np.array(states)).to(device)  # Move to device
        actions = torch.LongTensor(actions).to(device)  # Move to device
        rewards = torch.FloatTensor(rewards).to(device)  # Move to device
        next_states = torch.FloatTensor(np.array(next_states)).to(device)  # Move to device
        dones = torch.FloatTensor(dones).to(device)  # Move to device

        # Get current Q values for chosen actions
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Compute the expected Q values
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

def train(env, agent, episodes, batch_size=128, target_update=50,save_model_path='dqn_model.pth'):
    episode_rewards = []
    episode_losses = []
    data = {
        "Episode": [],
        "Total Reward": [],
        "Avg Loss": [],
        "Deployed Nodes": [],
        "Coverage": []
    }

    for episode in range(episodes):
        state = env.reset()
        #print_initial_R_N(state, env.n_potential_nodes)
        total_reward = 0
        total_loss = 0
        done = False
        step_count = 0
        step_details = []
        all_episodes_details = []

        while not done:
            action = agent.select_action(state)
            node_index = action // 2  # 根据你的动作设计来计算节点索引
            #print(f'Before action, node {node_index} state in flattened state: {state[node_index]}')  # 打印动作前的节点状态
            next_state, reward, done, _ = env.step(action)
            #print_changes_R_N(state, next_state, env.n_potential_nodes, action)  # Print changes
            #print(f'After action, node {node_index} state in flattened state: {next_state[node_index]}')  # 打印动作后的节点状态
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.experience_replay(batch_size)
            state = next_state
            #print(f'After action 1, node {node_index} state in flattened state: {state[node_index]}')  # 打印动作后的节点状态
            total_reward += reward
            #print(total_reward)
            total_loss += loss
            step_count += 1
            agent.total_steps += 1

            if episode % 20 == 0:
                coverage_percentage = env.calculate_coverage_percentage()
                env.plot_deployment(episode, coverage_percentage)
            # Log step details
            step_details.append({
                "Episode": episode,
                "Step": step_count,
                "Action": action,
                "Reward": reward,
                "Coverage": env.calculate_coverage_percentage()  # Assuming this method exists
            })
        if episode % target_update == 0:
            agent.update_target_network()

        avg_loss = total_loss / step_count if step_count else 0
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        all_episodes_details.extend(step_details)
        deployed_nodes = env.total_deployed_nodes()
        print("Episode",episode, "Deployed Nodes",deployed_nodes)
        coverage_percentage = env.calculate_coverage_percentage()
        data["Episode"].append(episode)
        data["Total Reward"].append(total_reward)
        data["Avg Loss"].append(avg_loss)
        data["Deployed Nodes"].append(deployed_nodes)
        print("Deployed 12", data["Deployed Nodes"])
        data["Coverage"].append(coverage_percentage)

        #agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay_episode)
        #env.plot_deployment()
        print(f"Episode {episode}: Total Reward = {total_reward}, Avg Loss = {avg_loss:.4f}, Deployed Nodes = {deployed_nodes}, Coverage = {coverage_percentage:.2f}%")


    avg_rewards_per_scale_episodes = [np.mean(episode_rewards[i:i+scale]) for i in range(0, len(episode_rewards),scale)]
    avg_losses_per_scale_episodes = [np.mean(episode_losses[i:i+scale]) for i in range(0, len(episode_losses), scale)]
    avg_numofnodes_per_scale_episodes = [np.mean(data["Deployed Nodes"][i:i + scale]) for i in range(0, len(data["Deployed Nodes"]), scale)]
    avg_coverage_per_scale_episodes = [np.mean(data["Coverage"][i:i + scale]) for i in
                                         range(0, len(data["Coverage"]), scale)]
    episodes_scale = list(range(0, len(episode_rewards), scale))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'all_episodes_step_details {timestamp}.csv'
    with open(filename, mode='w', newline='') as file:
        fieldnames = ["Episode", "Step", "Action", "Reward", "Coverage"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for detail in all_episodes_details:
            writer.writerow(detail)


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Reward per Episode')
    plt.plot(episodes_scale, avg_rewards_per_scale_episodes, label='Avg Reward per 10 Episodes', color='red', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Episode vs Reward')
    # Generate a timestamp or unique identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'Episode vs Reward_Time_{timestamp}.png'
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episode_losses, label='Loss per Episode')
    plt.plot(episodes_scale, avg_losses_per_scale_episodes, label='Avg Loss per 10 Episodes', color='blue', linewidth=2)
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
    plt.plot(episodes_scale, avg_numofnodes_per_scale_episodes, label='Avg nodes per 10 Episodes', color='red',
             linewidth=2)
    plt.title('Episode vs Number of Deployed Nodes removing all')
    plt.legend()

    # Plot Coverage Percentage vs. Episode
    plt.subplot(1, 2, 2)
    plt.plot(data["Episode"], data["Coverage"], label='Coverage Percentage per Episode', color='purple', marker='o',
             linestyle='-', linewidth=1, markersize=4)
    plt.plot(episodes_scale, avg_coverage_per_scale_episodes, label='Avg nodes per 10 Episodes', color='red',
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
    print("Initial R:", state[R_start:R_start + n_potential_nodes])
    print("Initial N row 0:", state[N_start:N_start + n_potential_nodes])  # Example: first row



class NetworkDeploymentEnv(gym.Env):
    def __init__(self):
        super(NetworkDeploymentEnv, self).__init__()
        self.grid_size = 100
        self.node_spacing = 20
        self.n_potential_nodes = (self.grid_size // self.node_spacing) ** 2
        self.action_space = spaces.MultiBinary(2 * self.n_potential_nodes)
        self.observation_space = spaces.Dict({
            "D": spaces.MultiBinary(self.n_potential_nodes),
            "R": spaces.Box(low=0, high=20, shape=(self.n_potential_nodes,)),
            "N": spaces.MultiBinary((self.n_potential_nodes, self.n_potential_nodes))
        })
        self.overhead = 1.2
        self.node_data_rate =2
        self.donor_data_rate = 15  # 15 Gbps
        self.coverage_radius = 20
        self.backhaul_radius = 30
        self.max_steps = max_steps
        self.current_step = 0
        self.reset()

    def step(self, action_index):
        rewards = 0
        done = False

        node_index = action_index // 2
        action_type = action_index % 2  # 0 for deploy, 1 for remove

        action_desc = "deploying" if action_type == 0 else "removing"
        print(f"Action: {action_desc} node at position {node_index}")

        if node_index in self.permanent_donors and action_type == 1:
            # Trying to remove a pre-fixed donor node
            print(f"Attempted to remove a pre-fixed donor at {node_index}. High penalty applied.")
            rewards = self.calculate_reward() * 4

        elif action_type == 0:
            rewards += self.deploy_node(node_index)
        else:
            rewards += self.remove_node(node_index)
        #self.print_network_status()
        total_reward = rewards
        self.current_step += 1

        if self.current_step >= self.max_steps:
            done = True
            self.current_step = 0

        return self._get_flattened_state(), total_reward, done, {}

    def deploy_node(self, node_index):
        if self.state["D"][node_index] == 1:  # 如果节点已经被部署
            print(f"Node {node_index} is already deployed. High penalty applied.")
            rewards = self.calculate_reward()*2
            return rewards  # 返回高惩罚
        self.state["D"][node_index] = 1  # 部署节点
        connected = self.reconnect_node(node_index)  # 尝试连接到现有网络
        if not connected:
            print(f"Deployed node {node_index} could not find a node to connect. High penalty applied.")
            self.state["D"][node_index] = 0
            rewards = self.calculate_reward() * 3
            return rewards  # 如果没有找到可以连接的节点，返回高惩罚
        print(f"Successfully deployed and connected node at index {node_index}.")
        rewards = self.calculate_reward()
        return rewards


    def remove_node(self, node_index):
        if self.state["D"][node_index] == 0:  # 如果节点位置上没有节点
            print(f"No node at position {node_index} to be removed.")
            rewards = self.calculate_reward()
            return rewards
        self.state["D"][node_index] = 0  # 移除节点
        # Mark the node for removal
        nodes_to_remove = [node_index]
        # Find all nodes connected to this node and mark them for removal as well
        for i in range(self.n_potential_nodes):
            if self.state["N"][node_index][i] == 1:
                if i not in self.permanent_donors:  # Ensure we don't remove donor nodes
                    nodes_to_remove.append(i)
        # Remove all marked nodes
        for idx in nodes_to_remove:
            self.state["D"][idx] = 0  # Remove the node
            # Reset connections for the removed node
            for j in range(self.n_potential_nodes):
                self.state["N"][idx][j] = 0
                self.state["R"][idx] = 0
        rewards = self.calculate_reward()
        return rewards

    def ensure_network_integrity(self):
        isolated_nodes = [i for i in range(self.n_potential_nodes) if self.state["D"][i] == 1 and not any(self.state["N"][i]) and not any(self.permanent_donors)]
        if isolated_nodes:
            return False  # 如果有孤立节点无法重新连接，返回False
        if not isolated_nodes:
            print("No isolated nodes found. Network integrity maintained.")
        return True  # 所有孤立节点都成功重新连接，返回True

    def print_network_status(self):
        print("Current Connection Matrix (N):")
        for i in range(self.n_potential_nodes):
            print(f"Node {i}: {self.state['N'][i]}")
        print("\nCurrent Data Rate Status (R):")
        for i in range(self.n_potential_nodes):
            print(f"Node {i}: Data Rate = {self.state['R'][i]}")
        print("\n")

    def reconnect_node(self, node_index):
        best_target = None
        max_data_rate = 2.4
        for i in range(self.n_potential_nodes):
            if self.state["D"][i] == 1 and i != node_index:
                distance = self.calculate_distance(i, node_index)
                if distance <= self.backhaul_radius and self.state["R"][i] > max_data_rate:
                    best_target = i
                    max_data_rate = self.state["R"][i]
        if best_target is not None:
            self.state["N"][best_target][node_index] = 1  # 根据网络模型可能需要调整为单向连接
            self.state["R"][best_target] -= self.node_data_rate * self.overhead
            # 遍历所有节点，找到与node_index直接连接的节点，并减少它们的data rate
            for i in range(self.n_potential_nodes):
                if self.state["N"][best_target][i] == 1:  # 如果best_target与i节点连接
                    self.state["R"][i] -= self.node_data_rate * self.overhead  # 调整data rate
            print(f"Reconnected node {node_index} to node {best_target} with data rate {self.state['R'][best_target]}")
            self.state["R"][node_index] = self.state["R"][best_target]
            return True
        return False

    def reset(self):
        self.state = {
            "D": np.zeros(self.n_potential_nodes),
            "R": np.zeros(self.n_potential_nodes),
            "N": np.zeros((self.n_potential_nodes, self.n_potential_nodes))
        }
        self.state["N"] = np.zeros((self.n_potential_nodes, self.n_potential_nodes))
        self.state["R"] = np.zeros(self.n_potential_nodes)
        # Randomly choose 10 donor locations
        self.permanent_donors = np.random.choice(range(self.n_potential_nodes), numberofdonor, replace=False)
        for idx in self.permanent_donors:
            self.state["D"][idx] = 1  # Mark as deployed
            self.state["R"][idx] = 15

        return self._get_flattened_state()

    def _get_flattened_state(self):
        # Flatten the state components into a single array
        D_flat = self.state["D"].flatten()
        R_flat = self.state["R"].flatten()
        N_flat = self.state["N"].flatten()
        flattened_state = np.concatenate([D_flat, R_flat, N_flat])
        #print(flattened_state)
        return flattened_state

    def render(self, mode='human'):
        pass

    def total_deployed_nodes(self):
        return np.sum(self.state["D"])-numberofdonor

    def calculate_reward(self):
        alpha = 10  # Penalty for uncovered area
        beta = 0.5  # Penalty for each deployed node
        #gamma = 100  # Reward multiplier for coverage

        total_area = self.grid_size * self.grid_size
        covered_addarea = len(self.calculate_addcoverage())
        covered_dorarea = len(self.calculate_dorcoverage())
        uncovered_area = total_area - covered_addarea- covered_dorarea

        #coverage_percentage = (covered_area / total_area) * 100

        # Calculate penalties and rewards
        uncovered_area_penalty = uncovered_area  * alpha
        deployment_penalty = beta * self.total_deployed_nodes()
        #coverage_reward = gamma * (covered_area / total_area)  # Reward based on the percentage of area covered

        # Final reward is thOld N row e coverage reward minus penalties
        reward = - uncovered_area_penalty - deployment_penalty
        return reward

    def calculate_addcoverage(self):
        # Calculate coverage by all nodes including predefined donors
        total_covered_grids = set()
        for i in range(self.n_potential_nodes):
            if self.state["D"][i] == 1:
                node_x, node_y = self.get_node_position(i)
                for x in range(node_x - self.coverage_radius, node_x + self.coverage_radius + 1):
                    for y in range(node_y - self.coverage_radius, node_y + self.coverage_radius + 1):
                        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                            total_covered_grids.add((x, y))

        # Calculate coverage by predefined donors only
        donor_covered_grids = set()
        for idx in self.permanent_donors:
            donor_x, donor_y = self.get_node_position(idx)
            for x in range(donor_x - self.coverage_radius, donor_x + self.coverage_radius + 1):
                for y in range(donor_y - self.coverage_radius, donor_y + self.coverage_radius + 1):
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        donor_covered_grids.add((x, y))

        # Calculate additional coverage provided by deployed nodes excluding predefined donors
        additional_coverage = total_covered_grids - donor_covered_grids
        return additional_coverage

    def calculate_dorcoverage(self):
        donor_covered_grids = set()
        for idx in self.permanent_donors:
            donor_x, donor_y = self.get_node_position(idx)
            for x in range(donor_x - self.coverage_radius, donor_x + self.coverage_radius + 1):
                for y in range(donor_y - self.coverage_radius, donor_y + self.coverage_radius + 1):
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        donor_covered_grids.add((x, y))
        return donor_covered_grids

    def calculate_coverage_percentage(self):
        total_area = self.grid_size * self.grid_size
        covered_dorarea = len(self.calculate_dorcoverage())
        additional_covered_area = len(self.calculate_addcoverage())
        coverage_percentage = (additional_covered_area +covered_dorarea/ total_area) * 100
        return coverage_percentage


    def get_node_position(self, node_index):
        row = node_index // (self.grid_size // self.node_spacing)
        col = node_index % (self.grid_size // self.node_spacing)
        x = col * self.node_spacing
        y = row * self.node_spacing
        return x, y

    def can_provide_service(self, node_index):
        node_x, node_y = self.get_node_position(node_index)
        for i in range(self.n_potential_nodes):
            if self.state["D"][i] == 1:
                donor_x, donor_y = self.get_node_position(i)
                distance = np.sqrt((node_x - donor_x) ** 2 + (node_y - donor_y) ** 2)
                if distance <= self.backhaul_radius and self.state["R"][i] >= self.node_data_rate * self.overhead:
                    return True
        return False

    def update_connections(self):
        for i in range(self.n_potential_nodes):
            if self.state["D"][i] == 1:  # If node i is deployed
                # Find the closest node that can provide service
                closest_node, min_distance = None, float('inf')
                for j in range(self.n_potential_nodes):
                    if i != j and self.state["D"][j] == 1:
                        distance = self.calculate_distance(i, j)
                        if distance <= self.backhaul_radius and self.state["R"][
                            j] >= self.node_data_rate * self.overhead:
                            if distance < min_distance:
                                closest_node, min_distance = j, distance
                # Connect to the closest node if found
                if closest_node is not None:
                    self.state["N"][i][closest_node] = 1

    def update_network_data_rate(self):
        # Adjust data rates based on connections
        for i in range(self.n_potential_nodes):
            if self.state["D"][i] == 1:
                for j in range(self.n_potential_nodes):
                    if self.state["N"][i][j] == 1:
                        # Reduce the data rate of the connected node
                        self.state["R"][j] -= self.node_data_rate * self.overhead
                        break  # Ensure only one connection affects the data rate

    def calculate_distance(self, node_index1, node_index2):
        x1, y1 = self.get_node_position(node_index1)
        x2, y2 = self.get_node_position(node_index2)
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def plot_deployment(self, episode, coverage_percentage):
        grid = np.zeros((self.grid_size//self.node_spacing , self.grid_size //self.node_spacing))
        for i in range(self.n_potential_nodes):
            row = i // (self.grid_size // self.node_spacing)
            col = i % (self.grid_size // self.node_spacing)
            grid[row, col] = self.state["D"][i]

        #plt.figure(figsize=(10, 10))
        #plt.imshow(grid, cmap='Greys', interpolation='nearest')
        #plt.title(f'Final Deployment - Episode {episode}, Coverage: {coverage_percentage:.2f}%')
        # Generate a timestamp or unique identifier
        #timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #filename = f'Final_Deployment_Episode_{episode}_Time_{timestamp}.png'
        #plt.savefig(filename)
        #plt.close()  # Close the plot to free memory

max_steps = 20
scale = 100
numberofdonor = 5
env = NetworkDeploymentEnv()
state_dim = len(env.reset())
action_dim = env.action_space.n
agent = Agent(state_dim, action_dim)
rewards = train(env, agent, episodes=10000)