from collections import deque
import random
import copy

from MDP import apply_action


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.used = set()

    def insert(self, route, next_state, reward):
        self.buffer.append((copy.deepcopy(route), next_state, reward))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        routes = [copy.deepcopy(b[0]) for b in batch]
        next_states = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        return routes, next_states, rewards

    def full_buffer(self, first_buffer_input, mdp):

        for _ in range(first_buffer_input):
            route = {
                "path": [mdp.depot_num],
                "capacity": 0.0,
                "total_distance": 0.0,
                "current_node": mdp.depot_num
            }
            
            route_len = random.randint(1, max(1, mdp.num_nodes // (mdp.num_cars * 2)))
            candidates = [
                n for n in range(mdp.num_nodes)
                if n not in self.used
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

                    self.insert(route, new_node, reward)

                    route = apply_action(route, new_node, mdp)
                    self.used.add(new_node)
                