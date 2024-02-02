import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_dqn(env, model, episodes, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = deque(maxlen=10000)
    epsilon = epsilon_start
    losses = []
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_counter = 0  # Initialize step counter for each episode

        while not done and step_counter < 100:  # Add condition for maximum steps
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = model(state_tensor).argmax().item()
            next_state, reward, done, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            step_counter += 1  # Increment step counter

            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = model(next_states).max(1)[0]
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, expected_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            epsilon = max(epsilon_end, epsilon_decay * epsilon)

        episode_rewards.append(total_reward)
        print(f"Episode: {episode}, Steps: {step_counter}, Total Deployed Nodes: {env.total_deployed_nodes()}, Reward: {total_reward}, Loss: {loss.item() if losses else 'N/A'}")

    # Plotting the training loss over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Episodes')
    plt.legend()
    plt.show()

    return losses, episode_rewards

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
        self.node_data_rate = 0.2  # 0.2 Gbps
        self.donor_data_rate = 15  # 15 Gbps
        self.coverage_radius = 200
        self.backhaul_radius = 300
        self.current_step = 0
        self.reset()

    def step(self, action):
        rewards = 0
        #print(f"Step {self.current_step + 1}:")
        for i in range(self.n_potential_nodes):
            action_type = "Deploy" if action[i] == 1 else "Remove"
            if action[i] == 1 and self.state["D"][i] == 0:
                # Attempt to deploy a node
                self.state["D"][i] = 1  # Temporarily update "D" state
                if self.can_provide_service(i):
                    self.update_connections()
                    self.update_network_data_rate()
                    #print(f"  {action_type} node at position {i}. Success. Updated connections.")
                else:
                    rewards -= 10  # Large penalty for attempting to deploy without service
                    #print(f"  {action_type} node at position {i}. Failed. No service provided.")
            elif action[i] == 0 and self.state["D"][i] == 1:
                # Attempt to remove a node
                self.state["D"][i] = 0
                self.update_connections()
                self.update_network_data_rate()
                #print(f"  {action_type} node at position {i}. Node removed. Updated connections.")

        reward = self.calculate_reward() + rewards
        self.current_step += 1
        done = self.current_step >= 100

        # Print connection matrix after each action
        print("  Connection Matrix:")
        print(self.state["N"])
        return self._get_flattened_state(), reward, done, {}

    def reset(self):
        self.state = {
            "D": np.zeros(self.n_potential_nodes),
            "R": np.zeros(self.n_potential_nodes),
            "N": np.zeros((self.n_potential_nodes, self.n_potential_nodes))
        }
        # Randomly choose 10 donor locations
        donor_indices = np.random.choice(range(self.n_potential_nodes), 10, replace=False)
        for idx in donor_indices:
            self.state["D"][idx] = 1
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
        alpha = 10
        beta = 0.01

        total_area = self.grid_size * self.grid_size
        covered_area = len(self.calculate_coverage())
        uncovered_area = total_area - covered_area

        uncovered_area_penalty = (uncovered_area / total_area) * alpha

        deployment_penalty = beta * np.sum(self.state["D"])

        reward = -uncovered_area_penalty - deployment_penalty
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

env = NetworkDeploymentEnv()
state_dim = len(env.reset())  # Get the size of the flattened state
action_dim = env.action_space.n
dqn_model = DQN(state_dim, action_dim)
losses = train_dqn(env, dqn_model, episodes=1000)

