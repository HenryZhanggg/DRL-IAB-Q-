import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Grid and network parameters
grid_size = 100
node_radius = 20
backhaul_radius = 30
interval = 10
state_size = grid_size * grid_size
lambda_values = [0.3, 0.5, 0.8, 1.5, 2, 3]  # Combined lambda values
action_size = len(lambda_values)
hidden_size = 128
learning_rate = 0.01
mu_e = 0.5
epsilon = 0.1
max_episodes = 1000
total_area = grid_size * grid_size
gamma = 0.99
data_rate_donor = 20
communication_overhead = 1.2
MAX_DONORS = 1
fixed_node_data_rate = 0.5  # Fixed data rate consumption for each node

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations + n_actions, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

class Environment:
    def __init__(self, grid_size, node_radius, total_area):
        self.grid_size = grid_size
        self.node_radius = node_radius
        self.total_area = total_area
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.covered_areas = set()
        self.deployed_nodes = set()
        self.potential_locations = set()
        self.donor_data_rate = data_rate_donor
        self.node_data_rates = {}  # Stores data rates for each node
        self.connection_matrix = np.zeros((grid_size, grid_size), dtype=int)

    def reset(self):
        self.state.fill(0)
        self.covered_areas.clear()
        self.deployed_nodes.clear()
        self.potential_locations.clear()
        self.donor_data_rate = data_rate_donor
        self.node_data_rates.clear()
        self.connection_matrix.fill(0)
        self.initialize_potential_locations()
        for _ in range(MAX_DONORS):
            self.deploy_random_donor()
        return self.state.flatten()

    def initialize_potential_locations(self):
        for x in range(0, self.grid_size, interval):
            for y in range(0, self.grid_size, interval):
                self.potential_locations.add((x, y))

    def deploy_random_donor(self):
        center_x, center_y = self.grid_size // 2, self.grid_size // 2
        self.deploy_donor(center_x, center_y)

    def deploy_donor(self, x, y):
        print(f"Attempting to deploy node at ({x}, {y})")
        self.state[x, y] = 1
        self.covered_areas.add((x, y))
        self.deployed_nodes.add((x, y))
        self.node_data_rates[(x, y)] = self.donor_data_rate
        self.update_potential_locations(x, y)

    def deploy_node(self, x, y):
        if not self.is_within_backhaul_radius(x, y):
            return False
        if (x, y) in self.deployed_nodes or (x, y) not in self.potential_locations:
            return False
        new_coverage = self.calculate_new_coverage(x, y)
        node_data_rate_consumption = fixed_node_data_rate * new_coverage
        updated_data_rate = self.donor_data_rate - node_data_rate_consumption
        if updated_data_rate < 0:
            return False
        for dx in range(-self.node_radius, self.node_radius + 1):
            for dy in range(-self.node_radius, self.node_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.state[nx, ny] = 1
                    self.covered_areas.add((nx, ny))
        self.deployed_nodes.add((x, y))
        self.node_data_rates[(x, y)] = updated_data_rate
        self.update_data_rates(x, y, node_data_rate_consumption)
        self.update_potential_locations(x, y)

        # Update connection matrix
        if len(self.deployed_nodes) > 1:
            last_node_x, last_node_y = list(self.deployed_nodes)[-2]
            self.connection_matrix[last_node_x, last_node_y] = 1
        return True

    def update_data_rates(self, x, y, node_data_rate):
        for nx, ny in self.deployed_nodes:
            if self.is_within_radius(x, y, nx, ny):
                if (nx, ny) == (grid_size // 2, grid_size // 2):
                    self.donor_data_rate -= node_data_rate * communication_overhead
                else:
                    self.node_data_rates[(nx, ny)] -= node_data_rate * communication_overhead

    def is_within_backhaul_radius(self, x, y):
        for nx, ny in self.deployed_nodes:
            if np.sqrt((x - nx) ** 2 + (y - ny) ** 2) <= backhaul_radius:
                return True
        return False

    def is_within_radius(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) <= self.node_radius

    def update_potential_locations(self, x, y):
        for dx in range(-self.node_radius, self.node_radius + 1, interval):
            for dy in range(-self.node_radius, self.node_radius + 1, interval):
                mx, my = x + dx, y + dy
                if 0 <= mx < self.grid_size and 0 <= my < self.grid_size:
                    self.potential_locations.add((mx, my))

    def step(self, agent):
        print("Starting step method")
        best_location = None
        best_value = -np.inf

        if len(self.deployed_nodes) == 0:
            search_area = [loc for loc in self.potential_locations if self.is_within_backhaul_radius(*loc)]
        else:
            search_area = self.potential_locations

        for potential_location in search_area:
            if potential_location not in self.deployed_nodes:
                x, y = potential_location
                state_tensor = torch.FloatTensor(self.state.flatten()).unsqueeze(0)
                action_index = agent.select_action(state_tensor, epsilon, self)
                lambda_val = lambda_values[action_index]

                new_coverage = self.calculate_new_coverage(x, y)
                node_data_rate_consumption = fixed_node_data_rate * new_coverage
                data_rate_left = self.donor_data_rate - node_data_rate_consumption
                if data_rate_left > 0:
                    value = new_coverage / data_rate_left * lambda_val
                    if value > best_value:
                        best_value = value
                        best_location = (x, y)

        node_deployed = False
        if best_location is not None:
            node_deployed = self.deploy_node(*best_location)

        coverage_requirement_met = len(self.covered_areas) >= 0.99 * self.total_area
        if coverage_requirement_met:
            reward = 100
            done = True
        else:
            reward = -1 * len(self.deployed_nodes) if node_deployed else 0
            done = False
        print(f"Step completed. Node deployed: {node_deployed}, Done: {done}")
        return self.state.flatten(), reward, done

    def calculate_new_coverage(self, x, y):
        new_coverage = 0
        for dx in range(-self.node_radius, self.node_radius + 1):
            for dy in range(-self.node_radius, self.node_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.state[nx, ny] == 0:
                    new_coverage += 1
        return new_coverage

class Agent:
    def __init__(self, state_size, action_size, learning_rate, gamma, mu_e):
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.mu_e = mu_e
        self.eligibility_trace = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

    def select_action(self, state, epsilon, env):
        state_tensor = torch.FloatTensor(state)
        lambda_val = np.random.choice(lambda_values)
        action_index = lambda_values.index(lambda_val)
        action_tensor = torch.zeros((1, action_size))
        action_tensor[0, action_index] = 1
        print("State tensor shape:", state_tensor.shape)
        print("Action tensor shape:", action_tensor.shape)
        combined_input = torch.cat((state_tensor, action_tensor), dim=1)
        print(f"4.2")
        q_values = self.model(combined_input)
        print(f"5")
        if np.random.rand() > epsilon:
            selected_action = q_values.max(1)[1].item()
            print(f"6")
        else:
            selected_action = np.random.randint(0, action_size)
            print(f"7")
        # Log the selected action
        print(f"Selected action index: {selected_action}, Lambda value: {lambda_values[selected_action]}")
        return selected_action

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.zeros((1, action_size))
        action_tensor[0, action] = 1
        combined_input = torch.cat((state_tensor, action_tensor), dim=1)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        next_combined_input = torch.cat((next_state_tensor, action_tensor), dim=1)
        q_values = self.model(combined_input)
        max_q_new_state = torch.max(self.model(next_combined_input)).item()
        td_error = reward + self.gamma * max_q_new_state - q_values[0, action].item()
        self.optimizer.zero_grad()
        q_values[0, action].backward()
        for name, param in self.model.named_parameters():
            self.eligibility_trace[name] = self.mu_e * self.gamma * self.eligibility_trace[name] + param.grad
            param.data += learning_rate * td_error * self.eligibility_trace[name]
        print(f"Update method called. Action: {action}, Reward: {reward}, Done: {done}")

env = Environment(grid_size, node_radius, total_area)
agent = Agent(state_size, action_size, learning_rate, gamma, mu_e)
epi_loss = []
epi_rewards = []

for episode in range(max_episodes):
    state = env.reset()
    done = False
    episode_loss = 0
    episode_reward = 0
    step_counter = 0
    print(f"Starting Episode {episode + 1}")

    while not done:
        action = agent.select_action(torch.FloatTensor(state).unsqueeze(0), epsilon, env)
        next_state, reward, done = env.step(agent)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        episode_loss += reward ** 2
        episode_reward += reward

        step_counter += 1
        if step_counter >= 1000:
            done = True

        print(f"Episode {episode + 1}, Step {step_counter}, Reward: {reward}")

    epi_loss.append(episode_loss)
    epi_rewards.append(episode_reward)
    print(f"Episode {episode + 1} completed with loss: {episode_loss}, Reward: {episode_reward}")
    print(f"Number of nodes deployed in Episode {episode + 1}: {len(env.deployed_nodes)}")
    print(f"Coordinates of deployed nodes in Episode {episode + 1}: {list(env.deployed_nodes)}")
    print("Connection Matrix:")
    print(env.connection_matrix)

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
