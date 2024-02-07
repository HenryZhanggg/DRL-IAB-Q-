import os

# Allow multiple OpenMP libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Your existing code follows
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import gym
from gym import spaces
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import copy
import csv

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
    def __init__(self, state_dim, action_dim, learning_rate=0.003, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DQN(state_dim, action_dim)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)

    def select_action(self, state):
        if random.random() < self.epsilon:
            action_index = np.random.randint(self.action_dim)
            return action_index
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            action_index = q_values.argmax().item()
            return action_index

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)  # Ensure actions are long integers
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

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


def train(env, agent, episodes, batch_size=64, target_update=10,save_model_path='dqn_model.pth'):
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
        total_reward = 0
        total_loss = 0
        done = False
        step_count = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.experience_replay(batch_size)
            state = next_state
            total_reward += reward
            total_loss += loss
            step_count += 1

        if episode % target_update == 0:
            agent.update_target_network()

        avg_loss = total_loss / step_count if step_count else 0
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        deployed_nodes = env.total_deployed_nodes()
        coverage_percentage = env.calculate_coverage_percentage()
        data["Episode"].append(episode)
        data["Total Reward"].append(total_reward)
        data["Avg Loss"].append(avg_loss)
        data["Deployed Nodes"].append(deployed_nodes)
        data["Coverage"].append(coverage_percentage)

        if agent.epsilon > agent.epsilon_end:
            agent.epsilon *= agent.epsilon_decay
            #print(f"Episode {episode}: Epsilon = {agent.epsilon:.4f}")  # print epsilon
        env.plot_deployment()
        print(f"Episode {episode}: Total Reward = {total_reward}, Avg Loss = {avg_loss:.4f}, Deployed Nodes = {deployed_nodes}, Coverage = {coverage_percentage:.2f}%")

    torch.save(agent.model.state_dict(), save_model_path)
    avg_rewards_per_100_episodes = [np.mean(episode_rewards[i:i+100]) for i in range(0, len(episode_rewards), 100)]
    avg_losses_per_100_episodes = [np.mean(episode_losses[i:i+100]) for i in range(0, len(episode_losses), 100)]
    episodes_100 = list(range(0, len(episode_rewards), 100))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Reward per Episode')
    plt.plot(episodes_100, avg_rewards_per_100_episodes, label='Avg Reward per 100 Episodes', color='red', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Episode vs Reward')
    plt.savefig('Episode vs Reward for all penalty reward.png')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episode_losses, label='Loss per Episode')
    plt.plot(episodes_100, avg_losses_per_100_episodes, label='Avg Loss per 100 Episodes', color='blue', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Episode vs Loss')
    plt.savefig('Episode vs Loss for all penalty reward.png')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # After the training loop, write data to a CSV file
    with open('training_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Episode', 'Total Reward', 'Avg Loss', 'Deployed Nodes', 'Coverage'])
        # Write the data
        for i in range(len(data['Episode'])):
            writer.writerow([data['Episode'][i], data['Total Reward'][i], data['Avg Loss'][i], data['Deployed Nodes'][i], data['Coverage'][i]])


    return episode_rewards, episode_losses

class NetworkDeploymentEnv(gym.Env):
    def __init__(self):
        super(NetworkDeploymentEnv, self).__init__()
        self.grid_size = 1000
        self.node_spacing = 50
        self.n_potential_nodes = (self.grid_size // self.node_spacing) ** 2
        self.action_space = spaces.MultiBinary(2 * self.n_potential_nodes)
        self.observation_space = spaces.Dict({
            "D": spaces.MultiBinary(self.n_potential_nodes),
            "R": spaces.Box(low=0, high=20, shape=(self.n_potential_nodes,)),
            "N": spaces.MultiBinary((self.n_potential_nodes, self.n_potential_nodes))
        })
        self.overhead = 1.2
        self.node_data_rate =5
        self.donor_data_rate = 15  # 15 Gbps
        self.coverage_radius = 200
        self.backhaul_radius = 300
        self.current_step = 0
        self.reset()

    def step(self, action_index):
        rewards = 0
        done = False

        # Determine the node index and action (deploy or remove) from the action index
        node_index = action_index // 2  # Assuming each node has two actions: deploy and remove
        action_type = action_index % 2  # 0 for deploy, 1 for remove

        # Perform the corresponding action
        if action_type == 0:  # Deploy action
            if self.state["D"][node_index] == 0:  # Check if the node is not already deployed
                self.state["D"][node_index] = 1  # Deploy the node
                print(f"Deployed node at position {node_index}.")
                if not self.can_provide_service(node_index):
                    rewards -= 100  # Penalty for deploying without service
                    #print(f"error1.")
                else:
                    self.update_connections()
                    self.update_network_data_rate()
                    #print(f"success1.")
            else:
                print(f"Position {node_index} is already deployed.")
        else:
            if self.state["D"][node_index] == 1:  # Check if the node is deployed
                self.state["D"][node_index] = 0  # Remove the node
                print(f"Removed node from position {node_index}.")
                # Update connections and network data rate after removing the node
                self.update_connections()
                self.update_network_data_rate()
            else:
                print(f"Position {node_index} is not deployed.")

        total_reward = self.calculate_reward() + rewards
        self.current_step += 1
        if self.current_step >= 100:  # Or any other condition to end the episode
            done = True
            self.current_step = 0

        return self._get_flattened_state(),total_reward,  done, {}

    def reset(self):
        self.state = {
            "D": np.zeros(self.n_potential_nodes),
            "R": np.zeros(self.n_potential_nodes),
            "N": np.zeros((self.n_potential_nodes, self.n_potential_nodes))
        }
        # Randomly choose 10 donor locations
        self.permanent_donors = np.random.choice(range(self.n_potential_nodes), 10, replace=False)
        for idx in self.permanent_donors:
            self.state["D"][idx] = 1  # Mark as deployed
        self.update_connections()
        self.update_network_data_rate()
        return self._get_flattened_state()

    def _get_flattened_state(self):
        # Flatten the state components into a single array
        D_flat = self.state["D"].flatten()
        R_flat = self.state["R"].flatten()
        N_flat = self.state["N"].flatten()
        flattened_state = np.concatenate([D_flat, R_flat, N_flat])
        return flattened_state

    def render(self, mode='human'):
        pass

    def total_deployed_nodes(self):
        return np.sum(self.state["D"])
    def calculate_reward(self):
        alpha = 10  # Penalty for uncovered area
        beta = 0.5  # Penalty for each deployed node
        gamma = 100  # Reward multiplier for coverage

        total_area = self.grid_size * self.grid_size
        covered_area = len(self.calculate_coverage())
        uncovered_area = total_area - covered_area

        coverage_percentage = (covered_area / total_area) * 100

        # Calculate penalties and rewards
        uncovered_area_penalty = (uncovered_area / total_area) * alpha
        deployment_penalty = beta * self.total_deployed_nodes()
        coverage_reward = gamma * (covered_area / total_area)  # Reward based on the percentage of area covered

        # Final reward is the coverage reward minus penalties
        reward = coverage_reward - uncovered_area_penalty - deployment_penalty
        return reward
    def calculate_reward(self):
        alpha = 100  # Penalty for uncovered area
        beta = 1  # Penalty for each deployed node

        total_area = self.grid_size * self.grid_size
        covered_area = len(self.calculate_coverage())
        uncovered_area = total_area - covered_area

        # Calculate penalties and rewards
        uncovered_area_penalty = uncovered_area * alpha
        deployment_penalty = self.total_deployed_nodes() * beta

        # Final reward is the negative sum of penalties
        reward = - uncovered_area_penalty - deployment_penalty
        return reward

    def calculate_coverage(self):
        covered_grids = set()
        for i in range(self.n_potential_nodes):
            if self.state["D"][i] == 1:
                node_x, node_y = self.get_node_position(i)
                for x in range(node_x - self.coverage_radius, node_x + self.coverage_radius + 1):
                    for y in range(node_y - self.coverage_radius, node_y + self.coverage_radius + 1):
                        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                            covered_grids.add((x, y))
        return covered_grids

    def calculate_coverage_percentage(self):
        total_area = self.grid_size * self.grid_size
        covered_area = len(self.calculate_coverage())
        coverage_percentage = (covered_area / total_area) * 100
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
            for j in range(self.n_potential_nodes):
                if i != j and self.state["D"][i] == 1:
                    distance = self.calculate_distance(i, j)
                    if distance <= self.backhaul_radius and self.state["R"][i] >= self.node_data_rate * self.overhead:
                        self.state["N"][i][j] = 1
                    else:
                        self.state["N"][i][j] = 0

    def update_network_data_rate(self):
        for i in range(self.n_potential_nodes):
            if self.state["D"][i] == 1:
                connected_nodes = [j for j in range(self.n_potential_nodes) if self.state["N"][i][j] == 1]
                total_data_rate_consumption = sum(self.node_data_rate * self.overhead for j in connected_nodes)
                self.state["R"][i] = max(0, self.donor_data_rate - total_data_rate_consumption)

    def calculate_distance(self, node_index1, node_index2):
        x1, y1 = self.get_node_position(node_index1)
        x2, y2 = self.get_node_position(node_index2)
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def plot_deployment(self):
        grid = np.zeros((self.grid_size // self.node_spacing, self.grid_size // self.node_spacing))
        for i in range(self.n_potential_nodes):
            row = i // (self.grid_size // self.node_spacing)
            col = i % (self.grid_size // self.node_spacing)
            grid[row, col] = self.state["D"][i]

        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='Greys', interpolation='nearest')
        plt.title('Final Deployment')
        plt.show()

env = NetworkDeploymentEnv()
state_dim = len(env.reset())
action_dim = env.action_space.n
agent = Agent(state_dim, action_dim)
model_save_path = 'dqn_network_deployment_model.pth'
rewards, losses = train(env, agent, episodes=1000, save_model_path=model_save_path)
