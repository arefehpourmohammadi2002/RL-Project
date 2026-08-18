import math
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

import replay_buffer as RB
from DQN_parent import DQN
from MDP import MDP


class SavingsQNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, marginal_cost_index):
        super().__init__()
        self.marginal_cost_index = marginal_cost_index
        self.residual = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, inputs):
        prior = -inputs[..., self.marginal_cost_index:self.marginal_cost_index + 1]
        return prior + self.residual(inputs)


class OnlyDQN(DQN):
    # 7 graph + 3 unused + 3 other-route + 3 current-route + 7 action features.
    INPUT_DIM = 23
    MARGINAL_COST_INDEX = 20

    def __init__(self, **parent_variables):
        input_dim = parent_variables.get("input_dim")
        if input_dim != self.INPUT_DIM:
            raise ValueError(f"OnlyDQN expects input_dim={self.INPUT_DIM}, got {input_dim}")
        super().__init__(**parent_variables)

        network_hidden_dim = max(64, parent_variables["hidden_dim"] * 2)
        self.explore_model = SavingsQNetwork(
            self.INPUT_DIM, network_hidden_dim, self.MARGINAL_COST_INDEX
        ).to(self.device)
        self.target_model = SavingsQNetwork(
            self.INPUT_DIM, network_hidden_dim, self.MARGINAL_COST_INDEX
        ).to(self.device)
        self.target_model.load_state_dict(self.explore_model.state_dict())

        self.lr = min(self.lr, 1e-3)
        self.optimizer = torch.optim.AdamW(
            self.explore_model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        self.criterion = nn.SmoothL1Loss()

    @staticmethod
    def distance_scale(mdp):
        distances = mdp.distance_matrix[~np.eye(mdp.num_nodes, dtype=bool)]
        return max(float(np.mean(distances)), 1e-6)

    @staticmethod
    def marginal_cost(mdp, route, next_node):
        current = route["current_node"]
        depot = mdp.depot_num
        return (
            mdp.distance_matrix[current][next_node]
            + mdp.distance_matrix[next_node][depot]
            - mdp.distance_matrix[current][depot]
        )

    def new_episode(self):
        num_nodes = random.randint(self.min_num_nodes, self.max_num_nodes)
        num_cars = random.randint(self.min_num_cars, self.max_num_cars)
        mdp = MDP(num_nodes, self.depot_num, num_cars, self.cars_capacity)
        mdp.build(
            self.min_distance,
            self.max_distance,
            self.min_node_dem,
            self.max_node_dem,
        )
        demands = [
            mdp.node_demand[i]
            for i in range(num_nodes)
            if i != mdp.depot_num
        ]
        if not self.can_pack_demands(
            demands, [mdp.cars_capacity] * num_cars
        ):
            return self.new_episode()

        self.num_nodes = num_nodes
        self.num_cars = num_cars
        return [self.new_route() for _ in range(num_cars)], mdp, {self.depot_num}

    @staticmethod
    def can_pack_demands(demands, capacities):
        items = sorted(
            (float(demand) for demand in demands), reverse=True
        )
        remaining = [float(capacity) for capacity in capacities]

        if not items:
            return True
        if not remaining or items[0] > max(remaining) + 1e-6:
            return False
        if sum(items) > sum(remaining) + 1e-6:
            return False

        for item in items:
            fitting_bins = [
                (capacity - item, index)
                for index, capacity in enumerate(remaining)
                if capacity + 1e-6 >= item
            ]
            if not fitting_bins:
                return False
            _space_after, best_index = min(fitting_bins)
            remaining[best_index] -= item

        return True

    def collect_candidates(self):
        candidate_map = {}
        empty_route_seen = False
        for route_idx, route in enumerate(self.routes):
            candidates = self.get_candidate(self.mdp, route, self.used)
            if not candidates:
                continue
            is_empty = len(route["path"]) == 1 and route["capacity"] == 0.0
            if is_empty and empty_route_seen:
                continue

            feasible_candidates = []
            for candidate in candidates:
                remaining_demands = [
                    self.mdp.node_demand[node]
                    for node in range(self.mdp.num_nodes)
                    if node not in self.used
                    and node != candidate
                    and node != self.mdp.depot_num
                ]
                remaining_caps = [
                    self.mdp.cars_capacity - other["capacity"]
                    - (self.mdp.node_demand[candidate] if idx == route_idx else 0.0)
                    for idx, other in enumerate(self.routes)
                ]
                if self.can_pack_demands(remaining_demands, remaining_caps):
                    feasible_candidates.append(candidate)

            if feasible_candidates:
                empty_route_seen = empty_route_seen or is_empty
                candidate_map[route_idx] = feasible_candidates
        return candidate_map

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

    def get_other_routes_static(self, routes, route):
        others = [other for other in routes if other is not route]
        if not others:
            return torch.zeros(3, device=self.device, dtype=torch.float32)
        distance_norm = max(
            (self.max_num_nodes - 1) * self.max_distance, self.max_distance
        )
        values = [
            sum(other["total_distance"] for other in others)
            / len(others)
            / distance_norm,
            sum(other["capacity"] for other in others)
            / len(others)
            / max(self.cars_capacity, 1e-6),
            sum(len(other["path"]) > 1 for other in others) / len(others),
        ]
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def get_current_route_statics(self, route):
        distance_norm = max(
            (self.max_num_nodes - 1) * self.max_distance, self.max_distance
        )
        load_fraction = route["capacity"] / max(self.cars_capacity, 1e-6)
        values = [
            route["total_distance"] / distance_norm,
            load_fraction,
            max(0.0, 1.0 - load_fraction),
        ]
        return torch.tensor(values, device=self.device, dtype=torch.float32)

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

    def training_episode(self, route_idx, next_node):
        route = self.routes[route_idx]
        reward = -self.marginal_cost(
            self.mdp, route, next_node
        ) / self.distance_scale(self.mdp)
        self.replay_buffer.insert(
            self.mdp, self.routes, route_idx, next_node, reward, self.used
        )

        self.explore_update_counter += 1
        if (
            self.explore_update_counter >= self.explore_update_limit
            and self.replay_buffer.size() >= self.batch_size
        ):
            self.train_step()
            self.explore_update_counter = 0


    def close_finished_routes(self, training):
        del training
        for route in self.routes:
            if (
                route["current_node"] != self.mdp.depot_num
                and not self.get_candidate(self.mdp, route, self.used)
            ):
                self.add_node_to_route(
                    self.mdp, route, self.mdp.depot_num, self.used
                )

    def prepare_replay_buffer(self):
        self.replay_buffer = RB.ReplayBuffer(capacity=self.RB_capacity)
        warmup_size = max(self.num_first_samples, self.batch_size)
        while self.replay_buffer.size() < warmup_size:
            self.run_random_episode(
                max_sample_of_one_env=max(4, self.batch_size // 4)
            )

    def DQN_train(self):
        self.prepare_replay_buffer()
        decay_epochs = max(int(self.num_epoches * 0.8), 1)
        epsilon_decay = (
            self.min_epsilon / max(self.epsilon, self.min_epsilon)
        ) ** (1 / decay_epochs)
        initial_lr = self.lr

        for epoch in range(self.num_epoches):
            self.run_episode(train=True)
            self.epsilon = max(self.min_epsilon, self.epsilon * epsilon_decay)
            self.epsilon_history.append(self.epsilon)

            progress = min((epoch + 1) / max(self.num_epoches, 1), 1.0)
            new_lr = self.min_lr + 0.5 * (initial_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
            for group in self.optimizer.param_groups:
                group["lr"] = new_lr
            self.lr_history.append(new_lr)

            if (epoch + 1) % max(self.num_epoches // 10, 1) == 0:
                recent = self.loss_history[-20:]
                mean_loss = sum(recent) / len(recent) if recent else float("nan")
                print(
                    f"epoch {epoch + 1:4d}/{self.num_epoches}  "
                    f"epsilon={self.epsilon:.3f}  loss={mean_loss:.4f}"
                )

    @staticmethod
    def route_distance(customers, mdp):
        path = [mdp.depot_num, *customers, mdp.depot_num]
        return sum(mdp.distance_matrix[a][b] for a, b in zip(path, path[1:]))
    def evaluate(self, mdp):
        return super().evaluate(mdp)

    
