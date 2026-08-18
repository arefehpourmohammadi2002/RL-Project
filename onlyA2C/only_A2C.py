import torch
import torch.nn as nn
import numpy as np
from A2C_parent import A2C
from MDP import MDP


class OnlyA2C(A2C):
    INPUT_DIM = 13

    def __init__(self, **parent_variables):
        input_dim = parent_variables.get("actor_input_dim")
        if input_dim is None:
            parent_variables["actor_input_dim"] = self.INPUT_DIM
        critic_dim = parent_variables.get("critic_input_dim")
        if critic_dim is None:
            parent_variables["critic_input_dim"] = 9
        super().__init__(**parent_variables)

    def get_graph_statics(self, mdp, train):
        distance_scale = self.distance_scale(mdp)
        mean_dist = np.mean(mdp.distance_matrix) / max(distance_scale, 1e-6)
        mean_dem = np.mean([mdp.node_demand[i] for i in range(mdp.num_nodes) if i != mdp.depot_num])
        mean_dem = mean_dem / max(mdp.cars_capacity, 1e-6)
        return torch.tensor([mean_dist, mean_dem, 0.0], device=self.device, dtype=torch.float32)

    def get_unused_nodes_statics(self, mdp, used):
        unused = [i for i in range(mdp.num_nodes) if i not in used]
        if not unused:
            return torch.zeros(3, device=self.device, dtype=torch.float32)
        return torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)

    def get_next_node_statics(self, mdp, route, next_node):
        dist = mdp.distance_matrix[route["current_node"]][next_node] / self.distance_scale(mdp)
        return torch.tensor([dist, 0.0, 0.0], device=self.device, dtype=torch.float32)

