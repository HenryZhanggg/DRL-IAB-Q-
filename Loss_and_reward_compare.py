import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
dqn_data = pd.read_csv('training_data_20240719-200947.csv')
ddqn_data = pd.read_csv('training_data_20240717-191205.csv')

# Ensure the columns exist in the data
required_columns = ['Episode', 'Total Reward', 'Avg Loss', 'Deployed Nodes', 'Coverage']
for column in required_columns:
    if column not in dqn_data.columns:
        raise ValueError(f"Column '{column}' not found in DQN data")
    if column not in ddqn_data.columns:
        raise ValueError(f"Column '{column}' not found in DDQN data")

window = 300
# Calculate the 800-episode moving averages
dqn_data['Avg Loss MA'] = dqn_data['Avg Loss'].rolling(window=window).mean()
ddqn_data['Avg Loss MA'] = ddqn_data['Avg Loss'].rolling(window=window).mean()
dqn_data['Total Reward MA'] = dqn_data['Total Reward'].rolling(window=window).mean()
ddqn_data['Total Reward MA'] = ddqn_data['Total Reward'].rolling(window=window).mean()
dqn_data['Deployed Nodes MA'] = dqn_data['Deployed Nodes'].rolling(window=window).mean()
ddqn_data['Deployed Nodes MA'] = ddqn_data['Deployed Nodes'].rolling(window=window).mean()
dqn_data['Coverage MA'] = dqn_data['Coverage'].rolling(window=window).mean()
ddqn_data['Coverage MA'] = ddqn_data['Coverage'].rolling(window=window).mean()

# Convert columns to numpy arrays
episodes_dqn = dqn_data['Episode'].values
episodes_ddqn = ddqn_data['Episode'].values

# Plot Loss vs Episodes (800 moving average)
plt.figure(figsize=(12, 6))
plt.plot(episodes_dqn, dqn_data['Avg Loss MA'].values, label='DQN Loss (800 MA)', color='blue')
plt.plot(episodes_ddqn, ddqn_data['Avg Loss MA'].values, label='DDQN Loss (800 MA)', color='green')
plt.xlabel('Episodes')
plt.ylabel('Average Loss (800 MA)')
plt.title('Loss vs Episodes (800 Moving Average)')
plt.ylim(0, 1000)  # Adjust the Y-axis scale as needed
plt.legend()
plt.savefig('Loss_vs_Episodes_Comparison_MA.png')
plt.close()

# Plot Reward vs Episodes (800 moving average)
plt.figure(figsize=(12, 6))
plt.plot(episodes_dqn, dqn_data['Total Reward MA'].values, label='DQN Reward (800 MA)', color='blue')
plt.plot(episodes_ddqn, ddqn_data['Total Reward MA'].values, label='DDQN Reward (800 MA)', color='green')
plt.xlabel('Episodes')
plt.ylabel('Total Reward (800 MA)')
plt.title('Reward vs Episodes (800 Moving Average)')
plt.ylim(-80000, 0)  # Adjust the Y-axis scale as needed
plt.legend()
plt.savefig('Reward_vs_Episodes_Comparison_MA.png')
plt.close()

# Plot Number of Deployed Nodes vs Episodes (800 moving average)
plt.figure(figsize=(12, 6))
plt.plot(episodes_dqn, dqn_data['Deployed Nodes MA'].values, label='DQN Deployed Nodes (800 MA)', color='blue')
plt.plot(episodes_ddqn, ddqn_data['Deployed Nodes MA'].values, label='DDQN Deployed Nodes (800 MA)', color='green')
plt.xlabel('Episodes')
plt.ylabel('Number of Deployed Nodes (800 MA)')
plt.title('Number of Deployed Nodes vs Episodes (800 Moving Average)')
plt.ylim(10, 23)  # Adjust the Y-axis scale as needed
plt.legend()
plt.savefig('Deployed_Nodes_vs_Episodes_Comparison_MA.png')
plt.close()

# Plot Coverage vs Episodes (800 moving average)
plt.figure(figsize=(12, 6))
plt.plot(episodes_dqn, dqn_data['Coverage MA'].values, label='DQN Coverage (800 MA)', color='blue')
plt.plot(episodes_ddqn, ddqn_data['Coverage MA'].values, label='DDQN Coverage (800 MA)', color='red')
plt.xlabel('Episodes')
plt.ylabel('Coverage (800 MA)')
plt.title('Coverage vs Episodes (800 Moving Average)')
plt.ylim(99, 100.1)  # Adjust the Y-axis scale to a larger scale
plt.legend()
plt.savefig('Coverage_vs_Episodes_Comparison_MA.png')
plt.close()
