import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Grid and network parameters
grid_size = 100
camera_interval = 10
camera_radius = 20
state_size = grid_size * grid_size
action_size = (grid_size // camera_interval) ** 2
hidden_size = 128
learning_rate = 0.01
alpha = 0.1
mu_e = 0.9
epsilon = 0.1
max_episodes = 10
total_area = grid_size * grid_size
gamma = 0.99  # Discount factor

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
    def __init__(self, grid_size, camera_interval, camera_radius, total_area):
        self.grid_size = grid_size
        self.camera_interval = camera_interval
        self.camera_radius = camera_radius
        self.total_area = total_area
        self.state = None
        self.covered_areas = set()
        self.deployed_cameras = set()  # 新增：用于跟踪已部署相机的位置
        self.potential_locations = set()

    def reset(self):
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.covered_areas.clear()
        self.deployed_cameras.clear()  # 重置已部署相机的位置
        self.potential_locations.clear()
        center_x, center_y = self.grid_size // 2, self.grid_size // 2
        self.deploy_camera(center_x, center_y)
        return self.state.flatten()

    def deploy_camera(self, x, y):
        if (x, y) in self.deployed_cameras:  # 检查该位置是否已部署相机
            return False  # 如果已部署，则不再部署

        self.deployed_cameras.add((x, y))  # 记录已部署相机的位置
        for dx in range(-self.camera_radius, self.camera_radius + 1):
            for dy in range(-self.camera_radius, self.camera_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.state[nx, ny] = 1
                    self.covered_areas.add((nx, ny))
        print(f"Camera deployed at ({x}, {y}). Total grids covered: {len(self.covered_areas)}")
        self.update_potential_locations(x, y)
        return True  # 返回成功部署的标志

    def update_potential_locations(self, x, y):
        for dx in range(-self.camera_radius, self.camera_radius + 1):
            for dy in range(-self.camera_radius, self.camera_radius + 1):
                mx, my = x + dx, y + dy
                if 0 <= mx < self.grid_size and 0 <= my < self.grid_size:
                    self.potential_locations.add((mx, my))

    def step(self, action):
        x, y = (action // (self.grid_size // self.camera_interval)) * self.camera_interval, (
                    action % (self.grid_size // self.camera_interval)) * self.camera_interval
        if (x, y) in self.deployed_cameras:
            return self.state.flatten(), -10, False  # Penalize for deploying at the same location

        # Check if the action is in potential locations
        if (x, y) not in self.potential_locations:
            return self.state.flatten(), 0, False

        new_coverage = 0
        # Update the state based on the action
        for dx in range(-self.camera_radius, self.camera_radius + 1):
            for dy in range(-self.camera_radius, self.camera_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.state[nx, ny] == 0:
                    self.state[nx, ny] = 1
                    new_coverage += 1
                    self.covered_areas.add((nx, ny))

        # Update potential locations for future actions
        self.update_potential_locations(x, y)

        # Calculate reward
        reward = -1 if new_coverage > 0 else 0
        # Check if the episode is done (95% coverage)
        done = len(self.covered_areas) >= 0.9 * self.total_area
        if done:
            reward += 10
        print(
            f"Step: Action taken at ({x}, {y}). Coverage: {len(self.covered_areas)}/{self.total_area} ({len(self.covered_areas) / self.total_area * 100:.2f}%)")
        return self.state.flatten(), reward, done

class Agent:
    def __init__(self, state_size, action_size, learning_rate, gamma, mu_e):
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.mu_e = mu_e
        self.eligibility_trace = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

    def select_action(self, state, epsilon, env, camera_interval, camera_radius):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        if np.random.rand() > epsilon:
            return self.select_nbv_action(state, env, camera_interval, camera_radius)
        else:
            return np.random.randint(action_size)

    def select_nbv_action(self, state, env, camera_interval, camera_radius):
        best_action = None
        max_coverage = 0

        for potential_location in env.potential_locations:
            if any([np.linalg.norm(np.array(potential_location) - np.array(covered_location)) <= 20 for covered_location in env.covered_areas]):
                x, y = potential_location
                action = (x // camera_interval) * (grid_size // camera_interval) + (y // camera_interval)
                new_coverage = self.calculate_new_coverage(x, y, state, camera_radius)
                if new_coverage > max_coverage:
                    max_coverage = new_coverage
                    best_action = action

        return best_action if best_action is not None else np.random.randint(action_size)

    def calculate_new_coverage(self, x, y, state, camera_radius):
        new_coverage = 0
        state_2d = state.reshape((grid_size, grid_size))
        # 优化循环逻辑
        for dx in range(-camera_radius, camera_radius + 1):
            for dy in range(-camera_radius, camera_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size and state_2d[nx, ny] == 0:
                    new_coverage += 1
        return new_coverage

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        q_values = self.model(state_tensor)
        max_q_new_state = torch.max(self.model(next_state_tensor)).item()
        td_error = reward + self.gamma * max_q_new_state - q_values[0, action].item()
        #td_error = reward - q_values[0, action].item()
        self.optimizer.zero_grad()
        q_values[0, action].backward()

        for name, param in self.model.named_parameters():
            self.eligibility_trace[name] = self.mu_e * self.gamma * self.eligibility_trace[name] + param.grad
            param.data += learning_rate * td_error * self.eligibility_trace[name]

# Training loop
env = Environment(grid_size, camera_interval, camera_radius, total_area)
agent = Agent(state_size, action_size, learning_rate, gamma, mu_e)

epi_loss = []
epi_rewards = []
for episode in range(max_episodes):
    episode_loss = 0
    episode_reward = 0
    state = env.reset()
    done = False
    print(f"Starting Episode {episode + 1}/{max_episodes}")
    step_count = 0
    while not done:
        action = agent.select_action(state, epsilon, env, camera_interval, camera_radius)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        episode_loss += reward ** 2
        episode_reward += reward
    step_count += 1
    if step_count % 100 == 0:
        print(f"  Episode {episode + 1}, Step {step_count}, Reward: {reward}")
    epi_loss.append(episode_loss)
    print(f"Episode {episode + 1} completed with loss: {episode_loss}")

# Plot training loss
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

