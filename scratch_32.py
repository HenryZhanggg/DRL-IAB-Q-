import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Grid and network parameters
grid_size = 100
node_radius = 20
state_size = grid_size * grid_size
lambda_values = [0, 0.5, 1, 2]
action_size = len(lambda_values)
hidden_size = 128
learning_rate = 0.01
mu_e = 0.9
epsilon = 0.1
max_episodes = 1000
total_area = grid_size * grid_size
gamma = 0.99
data_rate_donor = 15  # Data rate for the donor
communication_overhead = 1.2

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

class Environment:
    def __init__(self, grid_size, node_radius, total_area):
        self.grid_size = grid_size
        self.node_radius = node_radius
        self.total_area = total_area
        self.state = None
        self.covered_areas = set()
        self.deployed_nodes = set()
        self.potential_locations = set()
        self.donor_data_rate = data_rate_donor
        self.node_data_rates = {}  # Stores data rates for each node

    def reset(self):
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.covered_areas.clear()
        self.deployed_nodes.clear()
        self.potential_locations.clear()
        self.donor_data_rate = data_rate_donor
        self.node_data_rates.clear()
        self.initialize_potential_locations()
        self.deploy_random_donor()
        return self.state.flatten()

    def initialize_potential_locations(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.potential_locations.add((x, y))

    def deploy_random_donor(self):
        # Convert the set of tuples into a list of tuples and then choose one
        donor_location = list(self.potential_locations)
        donor_location = donor_location[np.random.randint(len(donor_location))]
        self.deploy_donor(*donor_location)

    def deploy_donor(self, x, y):
        self.state[x, y] = 1
        self.covered_areas.add((x, y))
        self.deployed_nodes.add((x, y))
        self.node_data_rates[(x, y)] = self.donor_data_rate
        self.update_potential_locations(x, y)

    def deploy_node(self, x, y):
        if (x, y) in self.deployed_nodes:
            return False
        node_deployed = False
        node_data_rate = np.random.uniform(1, 1.5)  # Random data rate consumption for the node
        for dx in range(-self.node_radius, self.node_radius + 1):
            for dy in range(-self.node_radius, self.node_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.state[nx, ny] = 1
                    self.covered_areas.add((nx, ny))
                    node_deployed = True

        if node_deployed:
            self.deployed_nodes.add((x, y))
            self.node_data_rates[(x, y)] = node_data_rate
            self.update_data_rates(x, y, node_data_rate)
            self.update_potential_locations(x, y)
            return True

        return False

    def update_data_rates(self, x, y, node_data_rate):
        # Update data rates for donor and nodes based on connectivity and overhead
        for nx, ny in self.deployed_nodes:
            if self.is_within_radius(x, y, nx, ny):
                if (nx, ny) == (grid_size // 2, grid_size // 2):  # Donor
                    self.donor_data_rate -= node_data_rate * communication_overhead
                else:
                    self.node_data_rates[(nx, ny)] -= node_data_rate * communication_overhead

    def is_within_radius(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) <= self.node_radius

    def update_potential_locations(self, x, y):
        for dx in range(-self.node_radius, self.node_radius + 1):
            for dy in range(-self.node_radius, self.node_radius + 1):
                mx, my = x + dx, y + dy
                if 0 <= mx < self.grid_size and 0 <= my < self.grid_size:
                    self.potential_locations.add((mx, my))

    def step(self, action):
        lambda_val = lambda_values[action]
        best_location = None
        best_value = -np.inf

        for potential_location in self.potential_locations:
            if potential_location not in self.deployed_nodes:
                x, y = potential_location
                new_coverage = self.calculate_new_coverage(x, y)
                connectivity_compliance = self.calculate_connectivity_compliance(x, y)
                value = new_coverage / connectivity_compliance ** lambda_val
                if value > best_value:
                    best_value = value
                    best_location = (x, y)

        if best_location is not None:
            self.deploy_node(*best_location)

        reward = -1 * len(self.deployed_nodes)
        done = len(self.covered_areas) >= 0.9 * self.total_area

        return self.state.flatten(), reward, done

    def calculate_new_coverage(self, x, y):
        new_coverage = 0
        for dx in range(-self.node_radius, self.node_radius + 1):
            for dy in range(-self.node_radius, self.node_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.state[nx, ny] == 0:
                    new_coverage += 1
        return new_coverage

    def calculate_connectivity_compliance(self, x, y):
        compliance = 0
        for nx, ny in self.deployed_nodes:
            if self.is_within_radius(x, y, nx, ny):
                if (nx, ny) == (grid_size // 2, grid_size // 2) and self.donor_data_rate > 0:
                    compliance += 1
                elif self.node_data_rates.get((nx, ny), 0) > 0:
                    compliance += 1
                else:
                    compliance += 10
            else:
                compliance += 10
        return compliance

class Agent:
    def __init__(self, state_size, action_size, learning_rate, gamma, mu_e):
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.mu_e = mu_e
        self.eligibility_trace = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

    def select_action(self, state, epsilon, env):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        if np.random.rand() > epsilon:
            return q_values.max(1)[1].item()
        else:
            return np.random.randint(action_size)

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        q_values = self.model(state_tensor)
        max_q_new_state = torch.max(self.model(next_state_tensor)).item()
        td_error = reward + self.gamma * max_q_new_state - q_values[0, action].item()
        self.optimizer.zero_grad()
        q_values[0, action].backward()

        for name, param in self.model.named_parameters():
            self.eligibility_trace[name] = self.mu_e * self.gamma * self.eligibility_trace[name] + param.grad
            param.data += learning_rate * td_error * self.eligibility_trace[name]

# Training loop
env = Environment(grid_size, node_radius, total_area)
agent = Agent(state_size, action_size, learning_rate, gamma, mu_e)
epi_loss = []
epi_rewards = []
for episode in range(max_episodes):
    state = env.reset()
    done = False
    episode_loss = 0
    episode_reward = 0

    while not done:
        action = agent.select_action(state, epsilon, env)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        episode_loss += reward ** 2
        episode_reward += reward

    epi_loss.append(episode_loss)
    epi_rewards.append(episode_reward)
    print(f"Episode {episode + 1} completed with loss: {episode_loss}, Reward: {episode_reward}")
    print(f"Number of nodes deployed in Episode {episode + 1}: {len(env.deployed_nodes)}")
    print(f"Coordinates of deployed nodes in Episode {episode + 1}: {list(env.deployed_nodes)}")

# Plot training loss and rewards
plt.figure(figsize=(10, 6))
plt.plot(range(max_episodes), epi_loss, label='Training Loss')
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.title('Training Loss Over Episodes')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(max_episodes), epi_rewards, label='Total Reward per Episode')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Reward vs. Episode')
plt.legend()
plt.show()
