import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# Grid and network parameters
grid_size = 1000
node_radius = 200
backhaul_radius = 300
interval = 20
state_size = grid_size * grid_size
lambda_values = [0.3, 0.5, 0.8, 1.5, 2, 3]  # Combined lambda values
action_size = len(lambda_values)
hidden_size = 128
learning_rate = 0.01
mu_e = 0.5
epsilon = 0.1
max_episodes = 1
total_area = grid_size * grid_size
gamma = 0.99
data_rate_donor = 20
communication_overhead = 1.2
MAX_DONORS = 10
fixed_node_data_rate = 0.0005

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_size + action_size, 256)
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
        self.search_area = set()
        self.donor_data_rate = data_rate_donor
        self.node_data_rates = {}
        self.connection_matrix = np.zeros((grid_size, grid_size), dtype=int)

    def reset(self):
        self.state.fill(0)
        self.covered_areas.clear()
        self.deployed_nodes.clear()
        self.search_area.clear()
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
                self.search_area.add((x, y))

    def deploy_random_donor(self):
        x = random.choice(range(0, self.grid_size, interval))
        y = random.choice(range(0, self.grid_size, interval))
        self.deploy_donor(x, y)
        #center_x, center_y = self.grid_size // 2, self.grid_size // 2
        #self.deploy_donor(center_x, center_y)

    def get_node_data_rates(self):
        return self.node_data_rates

    def get_deployed_nodes(self):
        return self.deployed_nodes

    def deploy_donor(self, x, y):
        print(f"Attempting to deploy node at ({x}, {y})")
        self.state[x, y] = 1
        self.covered_areas.add((x, y))
        self.deployed_nodes.add((x, y))
        self.node_data_rates[(x, y)] = self.donor_data_rate
        self.update_potential_locations(x, y)

    def deploy_node(self, x, y):
        if not self.is_within_backhaul_radius(x, y):
            print(f"Location ({x}, {y}) is not within backhaul radius.")
            return False
        if (x, y) in self.deployed_nodes:
            print(f"Location ({x}, {y}) already has a deployed node.")
            return False
        if (x, y) not in self.search_area:
            print(f"Location ({x}, {y}) is not in the search area.")
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
        return True

    def update_data_rates(self, x, y, node_data_rate_consumption):
        # Decrease the data rate of the donor if the new node is within its radius
        if self.is_within_backhaul_radius(x, y):
            self.donor_data_rate -= node_data_rate_consumption * communication_overhead
        # Decrease the data rates for all other connected nodes within node_radius
        for nx, ny in self.deployed_nodes:
            if self.is_within_radius(x, y, nx, ny) and (nx, ny) != (x, y):  # Exclude the newly deployed node
                self.node_data_rates[(nx, ny)] -= node_data_rate_consumption * communication_overhead

    def is_within_backhaul_radius(self, x, y):
        for nx, ny in self.deployed_nodes:
            if np.sqrt((x - nx) ** 2 + (y - ny) ** 2) <= backhaul_radius:
                return True
        return False

    def is_within_radius(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) <= self.node_radius

    def update_search_area(self):
        self.search_area.clear()
        for node in self.deployed_nodes:
            x, y = node
            for dx in range(-backhaul_radius, backhaul_radius, interval):
                for dy in range(-backhaul_radius, backhaul_radius, interval):
                    mx, my = x + dx, y + dy
                    if 0 <= mx < self.grid_size and 0 <= my < self.grid_size:
                        distance = np.sqrt((x - mx) ** 2 + (y - my) ** 2)
                        if distance <= backhaul_radius:
                            self.search_area.add((mx, my))
        print(f"update_search_area_result: {self.search_area}")

    def update_potential_locations(self, x, y):
        for dx in range(-backhaul_radius, backhaul_radius + 1, interval):
            for dy in range(-backhaul_radius, backhaul_radius + 1, interval):
                mx, my = x + dx, y + dy
                if 0 <= mx < self.grid_size and 0 <= my < self.grid_size:
                    self.search_area.add((mx, my))

    def step(self, agent, action):
        print("Starting step method")
        best_location = None
        best_value = 0
        print(best_value)
        # 更新 search_area
        self.update_search_area()
        # 选择动作
        lambda_val = action
        # 在 search_area 中找到最佳部署位置
        for potential_location in self.search_area:
            if potential_location not in self.deployed_nodes:
                x, y = potential_location
                new_coverage = self.calculate_new_coverage(x, y)
                node_data_rate_consumption = fixed_node_data_rate * new_coverage
                data_rate_left = self.donor_data_rate - node_data_rate_consumption
                if data_rate_left > 0:
                    value = new_coverage / (data_rate_left ** lambda_val)
                    print(value)
                    if value > best_value:
                        print("2")
                        best_value = value
                        best_location = (x, y)
        print(f"Step completed. Node deployed: {best_location}")
        # 部署节点
        node_deployed = False
        if best_location is not None:
            node_deployed = self.deploy_node(*best_location)
            print(f"Node deployed: {node_deployed}")

        # 检查是否满足覆盖要求
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
    def __init__(self, state_size, action_size, learning_rate, gamma, mu_e, initial_epsilon, epsilon_decay, min_epsilon):
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.mu_e = mu_e
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.e = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

    def decay_epsilon(self):
        """Apply exponential decay to the epsilon value."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def select_action(self, state):
        # Reshape state to [1, state_size]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Generate a random action for exploration
        random_action = np.random.randint(0, action_size)
        action_tensor = torch.zeros((1, action_size))
        action_tensor[0, random_action] = 1

        # Concatenate state and action tensors
        combined_input = torch.cat((state_tensor, action_tensor), dim=1)

        # Forward pass through the network
        q_values = self.model(combined_input)

        # Epsilon-greedy policy
        if np.random.rand() > self.epsilon:
            selected_action = q_values.max(1)[1].item()
        else:
            selected_action = random_action

        print(f"Selected action index: {selected_action}, Lambda value: {lambda_values[selected_action]}")
        return selected_action, action_tensor

    def update(self, state, action, action_tensor, reward, next_state, done, step_counter):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        combined_input = torch.cat((state_tensor, action_tensor), dim=1)
        next_combined_input = torch.cat((next_state_tensor, torch.zeros_like(action_tensor)), dim=1)
        # Get the current Q values and the Q values for the next state
        try:
            current_q_values = self.model(combined_input)
            next_q_values = self.model(next_combined_input).detach()
        except Exception as e:
            print("Error during model forward pass:", e)
            raise
        # Select the Q value for the taken action
        current_q_value = current_q_values[0, action]
        # Compute the TD target
        max_next_q_value = next_q_values.max(1)[0]
        td_target = reward + (self.gamma * max_next_q_value * (not done))
        # Compute the TD error
        td_error = td_target - current_q_value
        # Backpropagate the error
        self.optimizer.zero_grad()
        td_error.backward()
        # Update the eligibility trace and the weights
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.e[name] = self.mu_e * self.gamma * self.e[name] + param.grad
                param.data += learning_rate * td_error.detach() * self.e[name]
        # Reset if maximum steps reached
        if step_counter >= 1000:
            for name, param in self.model.named_parameters():
                param.data += learning_rate * self.e[name]
            self.reset_eligibility_trace()
            return True
        return False

    def reset_eligibility_trace(self):
        for name in self.e:
            self.e[name] = torch.zeros_like(self.e[name])

initial_epsilon = 1.0  # Start with a high epsilon value for more exploration
epsilon_decay = 0.995  # Exponential decay rate
min_epsilon = 0.01     # Minimum epsilon value
agent = Agent(state_size, action_size, learning_rate, gamma, mu_e, initial_epsilon, epsilon_decay, min_epsilon)
env = Environment(grid_size, node_radius, total_area)
epi_loss = []
epi_rewards = []

for episode in range(max_episodes):
    state = env.reset()
    done = False
    episode_loss = 0
    episode_reward = 0
    step_counter = 0

    while not done:
        action, action_tensor = agent.select_action(state)
        next_state, reward, done = env.step(agent, action)
        agent.update(state, action, action_tensor, reward, next_state, done, step_counter)
        state = next_state
        episode_loss += reward ** 2
        episode_reward += reward
        step_counter += 1

    # Apply exponential decay to epsilon after each episode
    agent.decay_epsilon()
    epi_loss.append(episode_loss)
    epi_rewards.append(episode_reward)
    deployed_nodes = env.get_deployed_nodes()
    print(f"Episode {episode + 1} completed. Deployed nodes: {deployed_nodes}")
    node_data_rates = env.get_node_data_rates()

    # Create a grid
    #grid = np.zeros((grid_size, grid_size))

    # Mark the deployed nodes on the grid and store their data rates
    #for node in deployed_nodes:
     #   x, y = node
     #   grid[x, y] = node_data_rates[node]

    # Plotting
    #plt.figure(figsize=(10, 10))
    #plt.imshow(grid, cmap='hot', interpolation='nearest')
    #plt.colorbar(label='Data Rate Left')
    #plt.title('Deployed Nodes and Their Data Rates')
    #plt.xlabel('X Coordinate')
    #plt.ylabel('Y Coordinate')

    # Annotate each node with its data rate
    #for (x, y), data_rate in node_data_rates.items():
        #plt.text(y, x, f'{data_rate:.2f}', ha='center', va='center', color='blue')

   #plt.show()
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
