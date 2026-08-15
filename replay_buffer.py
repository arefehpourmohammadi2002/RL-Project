from collections import deque
from copy import deepcopy
import random

class SolutionSnapshot:
    def __init__(self, mdp, routes, chosen_route, next_node, reward, used):
        self.mdp = mdp
        self.routes = deepcopy(routes)
        self.chosen_route = deepcopy(chosen_route)
        self.next_node = next_node
        self.reward = reward
        self.used = used 

class ReplayBuffer:
    def __init__(self, capacity):
        self.replay_buffer = deque(maxlen=capacity)

    def insert(self, mdp, routes, chosen_route, next_node, reward, used):
        obj_solu_snap = SolutionSnapshot(mdp, routes, chosen_route, next_node, reward, used)
        self.replay_buffer.append(obj_solu_snap)

    def sample(self, batch_size):
        samples = random.sample(self.replay_buffer, batch_size)
        return samples

    def full_replay_buffer(self, num_first_samples):
        pass