from collections import deque
import random
import copy
import numpy as np
from MDP import MDP, apply_action
import GNN_embedding as GNN



class ReplayBuffer:
    def __init__(self, capacity, min_num_nodes, max_num_nodes, min_num_cars,
                 max_num_cars, cars_capacity, min_dis, max_dis, min_node_cap, 
                 max_node_cap, large_value, device):
        self.buffer = deque(maxlen=capacity)

        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.min_num_cars = min_num_cars
        self.max_num_cars = max_num_cars
        self.cars_capacity = cars_capacity
        self.min_dis = min_dis
        self.max_dis = max_dis
        self.min_node_cap = min_node_cap
        self.max_node_cap = max_node_cap
        self.large_value = large_value
        self.device = device

    def insert(self, route, next_state, reward, used_snapshot, mdp, gnn_input):

        self.buffer.append((
            copy.deepcopy(route), next_state, reward,
            frozenset(used_snapshot), mdp, gnn_input
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        routes = [copy.deepcopy(b[0]) for b in batch]
        next_states = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        used_sets = [b[3] for b in batch]
        mdps = [b[4] for b in batch]
        gnn_inputs = [b[5] for b in batch]
        return routes, next_states, rewards, used_sets, mdps, gnn_inputs

    def generate_random_env(self):
        num_nodes = np.random.randint(self.min_num_nodes, self.max_num_nodes + 1)
        num_cars = np.random.randint(self.min_num_cars, self.max_num_cars + 1)
        cars_capacity = np.random.uniform(1, self.cars_capacity)

        mdp = MDP(num_nodes, self.depot_num, num_cars, cars_capacity)
        mdp.fill_distance_matrix(self.min_dis, self.max_dis)
        mdp.fill_node_cap_matrix(self.min_node_cap, self.max_node_cap)

        node_feature = GNN.create_node_feature(mdp)
        edge_feature = GNN.create_edge_feature(mdp)
        gnn_input = GNN.create_input(node_feature, edge_feature, self.large_value)
        gnn_input = gnn_input.to(self.device)

        return mdp, gnn_input
    
    def full_buffer(self, first_buffer_input):

        for _ in range(first_buffer_input):
            mdp, gnn_input = self.env_generator()

            route = {
                "path": [mdp.depot_num],
                "capacity": 0.0,
                "total_distance": 0.0,
                "current_node": mdp.depot_num
            }
            local_used = set()

            route_len = random.randint(1, max(1, mdp.num_nodes // (mdp.num_cars * 2)))
            candidates = [
                n for n in range(mdp.num_nodes)
                if n not in local_used
                and n != mdp.depot_num
                and route["capacity"] + mdp.node_capacity[n] <= mdp.cars_capacity
            ]

            for _ in range(route_len):
                if not candidates:
                    break

                new_node = random.choice(candidates)
                candidates.remove(new_node)

                if route["capacity"] + mdp.node_capacity[new_node] <= mdp.cars_capacity:
                    distance = mdp.distance_matrix[route["current_node"], new_node]
                    reward = -1.0 * distance

                    self.insert(route, new_node, reward, local_used, mdp, gnn_input)

                    route = apply_action(route, new_node, mdp)
                    local_used.add(new_node)