import math
import random
from copy import deepcopy
from itertools import chain
import numpy as np
import torch

import replay_buffer as RB
from DQN_parent import DQN
from MDP import MDP


class OnlyDQN(DQN):
    # State  = 7 graph + 3 unused + 2 other-route (parent default) = 12.
    # Action = 7 next-node + 2 current-route + 6 shared route-action = 15.
    INPUT_DIM = 27
    STATE_DIM = 12
    ACTION_DIM = 15

    def __init__(self, **parent_variables):
        input_dim = parent_variables.get("input_dim")
        if input_dim != self.INPUT_DIM:
            raise ValueError(f"OnlyDQN expects input_dim={self.INPUT_DIM}, got {input_dim}")
        hidden_dim = parent_variables["hidden_dim"]
        super().__init__(
            state_net_input_dim=self.STATE_DIM,
            state_net_hidden_dim=hidden_dim,
            state_net_output_dim=hidden_dim,
            action_net_input_dim=self.ACTION_DIM,
            action_net_hidden_dim=hidden_dim,
            action_net_output_dim=hidden_dim,
            **parent_variables,
        )

        self.lr = min(self.lr, 1e-3)
        self.optimizer = torch.optim.AdamW(
           chain(self.explore_model.parameters(),
                self.action_net.parameters(),
                self.state_net.parameters()
                ),
             lr=self.lr
        )

    def get_graph_statics(self, mdp, train):
        del train
        scale = self.distance_scale(mdp)
        distances = mdp.distance_matrix[~np.eye(mdp.num_nodes, dtype=bool)]
        total_demand = float(np.sum(mdp.node_demand))
        fleet_capacity = max(mdp.num_cars * mdp.cars_capacity, 1e-6)
        values = [
            mdp.num_nodes / max(self.max_num_nodes, 1),
            mdp.num_cars / max(self.max_num_cars, 1),
            float(np.mean(distances)) / scale,
            float(np.std(distances)) / scale,
            mdp.cars_capacity / max(total_demand, 1e-6),
            mdp.node_dem_ave / max(mdp.cars_capacity, 1e-6),
            total_demand / fleet_capacity,
        ]
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def get_unused_nodes_statics(self, mdp, used):
        unused = self.get_unused_nodes(mdp, used)
        unused_demand = float(sum(mdp.node_demand[i] for i in unused))
        values = [
            len(unused) / max(mdp.num_nodes - 1, 1),
            unused_demand / len(unused) / mdp.cars_capacity if unused else 0.0,
            unused_demand / max(mdp.num_cars * mdp.cars_capacity, 1e-6),
        ]
        return torch.tensor(values, device=self.device, dtype=torch.float32)


    def marginal_cost(self, mdp, route, next_node):
        current = route["current_node"]
        depot = mdp.depot_num
        return (
            mdp.distance_matrix[current][next_node]
            + mdp.distance_matrix[next_node][depot]
            - mdp.distance_matrix[current][depot]
        )

    def get_next_node_statics(self, mdp, route, next_node):
        scale = self.distance_scale(mdp)
        current = route["current_node"]
        depot = mdp.depot_num
        current_distance = mdp.distance_matrix[current][next_node]
        depot_distance = mdp.distance_matrix[depot][next_node]
        marginal_cost = self.marginal_cost(mdp, route, next_node)
        saving = mdp.distance_matrix[current][depot] + depot_distance - current_distance
        values = [
            float(np.sum(mdp.distance_matrix[:, next_node]))
            / max((mdp.num_nodes - 1) * scale, scale),
            mdp.node_demand[next_node] / max(mdp.cars_capacity, 1e-6),
            current_distance / scale,
            depot_distance / scale,
            marginal_cost / scale,
            saving / scale,
            (
                mdp.cars_capacity
                - route["capacity"]
                - mdp.node_demand[next_node]
            )
            / max(mdp.cars_capacity, 1e-6),
        ]
        return torch.tensor(values, device=self.device, dtype=torch.float32)
