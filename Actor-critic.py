import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

# Actor Model
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# Critic Model
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# Network Deployment Environment
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
        for i in range(self.n_potential_nodes):
            if action[i] == 1:  # Deploy action
                if self.state["D"][i] == 0 and self.can_provide_service(i, True):
                    self.state["D"][i] = 1
                    self.update_connections()
                    self.update_network_data_rate()

            elif action[i + self.n_potential_nodes] == 1:  # Remove action
                if self.state["D"][i] == 1:
                    self.state["D"][i] = 0
                    self.update_connections()
                    self.update_network_data_rate()

        reward = self.calculate_reward()
        self.current_step += 1
        done = self.current_step >= 150
        return self.state, reward, done, {}

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

    def calculate_reward(self):
        covered_grids = self.calculate_coverage()
        penalty_uncovered = 0.1 * (self.grid_size * self.grid_size - len(covered_grids))
        penalty_node_deployment = 0.01 * np.sum(self.state["D"])
        return -penalty_uncovered - penalty_node_deployment

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

    def can_provide_service(self, node_index, deploy):
        node_x, node_y = self.get_node_position(node_index)
        for i in range(self.n_potential_nodes):
            if self.state["D"][i] == 1:
                donor_x, donor_y = self.get_node_position(i)
                distance = np.sqrt((node_x - donor_x)**2 + (node_y - donor_y)**2)
                if distance <= self.backhaul_radius and self.state["R"][i] >= self.node_data_rate * self.overhead:
                    return True
        return False if deploy else True

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

def train_actor_critic(env, actor, critic, episodes, actor_lr=0.001, critic_lr=0.001, gamma=0.99):
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    episode_rewards = []
    episode_losses = []
    episode_coverages = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_loss = []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = actor(state_tensor)
            action_dist = torch.distributions.Bernoulli(action_probs)
            action = action_dist.sample().numpy().astype(int)

            next_state, reward, done, _ = env.step(action.flatten())
            next_state_flattened = np.concatenate([next_state["D"].flatten(), next_state["R"].flatten(), next_state["N"].flatten()])

            value = critic(state_tensor)
            next_state_tensor = torch.FloatTensor(next_state_flattened).unsqueeze(0)
            next_value = critic(next_state_tensor)
            td_target = reward + gamma * next_value * (1 - int(done))
            td_error = td_target - value

            critic_loss = td_error.pow(2)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()

            actor_loss = -torch.sum(torch.log(action_probs) * action + torch.log(1 - action_probs) * (1 - action)) * td_error.detach()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state_flattened
            total_reward += reward
            episode_loss.append(critic_loss.item())

        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(episode_loss))

        # Calculate and store coverage percentage
        coverage_percentage = len(env.calculate_coverage()) / (env.grid_size * env.grid_size) * 100
        episode_coverages.append(coverage_percentage)
        print(
            f"Episode {episode} finished. Total Reward: {total_reward}, Coverage: {coverage_percentage:.2f}%, Deployment: {np.sum(next_state['D'])}")
    # Plotting
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title("Episode vs Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.savefig('Episode vs Rewards')

    plt.subplot(1, 3, 2)
    plt.plot(episode_losses)
    plt.title("Episode vs Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig("Episode vs Loss")

    plt.subplot(1, 3, 3)
    plt.plot(episode_coverages)
    plt.title("Episode vs Coverage Percentage")
    plt.xlabel("Episode")
    plt.ylabel("Coverage (%)")
    plt.savefig("Episode vs Coverage Percentage")
    plt.show()

# Main Execution
env = NetworkDeploymentEnv()
state_dim = len(env.reset())
action_dim = env.action_space.n

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
train_actor_critic(env, actor, critic, episodes=2000)