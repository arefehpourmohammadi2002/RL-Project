from collections import deque
import random
import copy

import numpy as np

from MDP import MDP, apply_action


class ReplayBuffer:
    def __init__(self, capacity, min_num_nodes, max_num_nodes, min_num_cars,
                 max_num_cars, cars_capacity, min_dis, max_dis, min_node_cap,
                 max_node_cap, device, depot_num):
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
        self.device = device
        self.depot_num = depot_num

    def insert(self, routes_snapshot, acting_key, action, reward, used_snapshot, mdp):
        # BUGFIX (carried over): stores every route's state at this moment,
        # not just the single acting route -- needed so train_step can later
        # compute the TRUE global-best next action across all routes, not
        # just the acting route in isolation. gnn_input is no longer stored
        # at all -- there's no GNN in this design, so nothing needs it.
        self.buffer.append((
            copy.deepcopy(routes_snapshot), acting_key, action, reward,
            frozenset(used_snapshot), mdp
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        routes_snapshots = [copy.deepcopy(b[0]) for b in batch]
        acting_keys = [b[1] for b in batch]
        actions = [b[2] for b in batch]
        rewards = [b[3] for b in batch]
        used_sets = [b[4] for b in batch]
        mdps = [b[5] for b in batch]
        return routes_snapshots, acting_keys, actions, rewards, used_sets, mdps

    def generate_random_env(self):
        num_nodes = np.random.randint(self.min_num_nodes, self.max_num_nodes + 1)
        num_cars = np.random.randint(self.min_num_cars, self.max_num_cars + 1)
        cars_capacity = np.random.uniform(1, self.cars_capacity)

        mdp = MDP(num_nodes, self.depot_num, num_cars, cars_capacity)
        mdp.fill_distance_matrix(self.min_dis, self.max_dis)
        mdp.fill_node_cap_matrix(self.min_node_cap, self.max_node_cap)

        return mdp

    def full_buffer(self, first_buffer_input):

        for _ in range(first_buffer_input):
            mdp = self.generate_random_env()

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

                    # Only one synthetic route exists here, so it's the only
                    # thing available to bootstrap against -- real experience
                    # collected during actual training episodes will supply
                    # proper multi-route context and quickly dominate the
                    # buffer over this warm-start data.
                    self.insert({"route0": route}, "route0", new_node, reward, local_used, mdp)

                    route = apply_action(route, new_node, mdp)
                    local_used.add(new_node)
