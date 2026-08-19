from collections import deque
from copy import deepcopy
import random


class SolutionSnapshot:
    def __init__(self, mdp, routes, route_idx, next_node, reward, used):
        self.mdp = mdp
        self.routes = deepcopy(routes)
        self.route_idx = route_idx
        self.next_node = next_node
        self.reward = reward
        self.used = deepcopy(used)


class ReplayBuffer:
    def __init__(self, capacity):
        self.replay_buffer = deque(maxlen=capacity)

    def insert(self, mdp, routes, route_idx, next_node, reward, used):
        obj_solu_snap = SolutionSnapshot(mdp=mdp, routes=routes, route_idx=route_idx,
                                         next_node=next_node, reward=reward, used=used)
        self.replay_buffer.append(obj_solu_snap)

    def sample(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)

    def size(self):
        return len(self.replay_buffer)
