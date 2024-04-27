import math
import pickle

def precompute_node_data(grid_size, node_spacing, coverage_radius, backhaul_radius):
    n_nodes_per_row = grid_size // node_spacing
    n_nodes = n_nodes_per_row ** 2
    node_positions = [(x * node_spacing, y * node_spacing) for y in range(n_nodes_per_row) for x in range(n_nodes_per_row)]

    coverage_map = {}
    connections_map = {}

    for i, pos_i in enumerate(node_positions):
        coverage = []
        connections = []

        # Compute coverage area for node i
        for x in range(pos_i[0] - coverage_radius, pos_i[0] + coverage_radius + 1):
            for y in range(pos_i[1] - coverage_radius, pos_i[1] + coverage_radius + 1):
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    #if math.sqrt((x - pos_i[0]) ** 2 + (y - pos_i[1]) ** 2) <= coverage_radius:
                    coverage.append((x, y))
        coverage_map[i] = coverage

        # Compute potential connections for node i
        for j, pos_j in enumerate(node_positions):
            if i != j:
                distance = math.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2)
                if distance <= backhaul_radius:
                    connections.append(j)
        connections_map[i] = connections

    # Save the precomputed data
    with open('coverage_map.pkl1002', 'wb') as f:
        pickle.dump(coverage_map, f)
    with open('connections_map.pkl1002', 'wb') as f:
        pickle.dump(connections_map, f)

    print("Pre-computation completed and saved.")

# Define parameters
grid_size = 1000
node_spacing = 50
coverage_radius = 200  # Coverage radius in grid units
backhaul_radius = 300  # Connection radius in grid units

# Run pre-computation
precompute_node_data(grid_size, node_spacing, coverage_radius, backhaul_radius)
