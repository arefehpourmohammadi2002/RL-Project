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
    """Q-network initialized with an objective-aligned insertion policy."""

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
    def _distance_scale(mdp):
        distances = mdp.distance_matrix[~np.eye(mdp.num_nodes, dtype=bool)]
        return max(float(np.mean(distances)), 1e-6)

    @staticmethod
    def _marginal_cost(mdp, route, next_node):
        """Increase in route length while preserving its implicit depot return."""
        current = route["current_node"]
        depot = mdp.depot_num
        return (
            mdp.distance_matrix[current][next_node]
            + mdp.distance_matrix[next_node][depot]
            - mdp.distance_matrix[current][depot]
        )

    def new_episode(self):
        """Generate a feasible instance instead of training on impossible VRPs."""
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
        if not self._can_pack_demands(
            demands, [mdp.cars_capacity] * num_cars
        ):
            return self.new_episode()

        self.num_nodes = num_nodes
        self.num_cars = num_cars
        return [self.new_route() for _ in range(num_cars)], mdp, {self.depot_num}

    @staticmethod
    def _can_pack_demands(demands, capacities):
        """Fast conservative packing check used to keep actions feasible."""
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
        """Collapse vehicle symmetry and reject actions that strand demand."""
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
                if self._can_pack_demands(remaining_demands, remaining_caps):
                    feasible_candidates.append(candidate)

            if feasible_candidates:
                empty_route_seen = empty_route_seen or is_empty
                candidate_map[route_idx] = feasible_candidates
        return candidate_map

    def get_graph_statics(self, mdp, train):
        del train
        scale = self._distance_scale(mdp)
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
        scale = self._distance_scale(mdp)
        current = route["current_node"]
        depot = mdp.depot_num
        current_distance = mdp.distance_matrix[current][next_node]
        depot_distance = mdp.distance_matrix[depot][next_node]
        marginal_cost = self._marginal_cost(mdp, route, next_node)
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
        reward = -self._marginal_cost(
            self.mdp, route, next_node
        ) / self._distance_scale(self.mdp)
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
        # Marginal-cost rewards already include the changing depot-return edge.
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
    def _route_distance(customers, mdp):
        path = [mdp.depot_num, *customers, mdp.depot_num]
        return sum(
            mdp.distance_matrix[a][b] for a, b in zip(path, path[1:])
        )

    def _local_search(self, routes, mdp):
        """Improve DQN routes with capacity-safe 2-opt, relocate, and swap."""
        customer_routes = [
            [node for node in route["path"] if node != mdp.depot_num]
            for route in routes
        ]
        loads = [
            sum(mdp.node_demand[node] for node in route)
            for route in customer_routes
        ]
        tolerance = 1e-10

        improved = True
        while improved:
            improved = False

            for route in customer_routes:
                old_distance = self._route_distance(route, mdp)
                best_delta, best_pair = 0.0, None
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        candidate = (
                            route[:i]
                            + list(reversed(route[i:j + 1]))
                            + route[j + 1:]
                        )
                        delta = self._route_distance(candidate, mdp) - old_distance
                        if delta < best_delta - tolerance:
                            best_delta, best_pair = delta, (i, j)
                if best_pair is not None:
                    i, j = best_pair
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True

            best_delta, best_move = 0.0, None
            for source_idx, source in enumerate(customer_routes):
                for node_pos, node in enumerate(source):
                    demand = mdp.node_demand[node]
                    for target_idx, target in enumerate(customer_routes):
                        if (
                            source_idx == target_idx
                            or loads[target_idx] + demand
                            > mdp.cars_capacity + tolerance
                        ):
                            continue
                        old = (
                            self._route_distance(source, mdp)
                            + self._route_distance(target, mdp)
                        )
                        new_source = source[:node_pos] + source[node_pos + 1:]
                        for insert_pos in range(len(target) + 1):
                            new_target = (
                                target[:insert_pos]
                                + [node]
                                + target[insert_pos:]
                            )
                            delta = (
                                self._route_distance(new_source, mdp)
                                + self._route_distance(new_target, mdp)
                                - old
                            )
                            if delta < best_delta - tolerance:
                                best_delta = delta
                                best_move = (
                                    source_idx,
                                    node_pos,
                                    target_idx,
                                    insert_pos,
                                    demand,
                                )
            if best_move is not None:
                source_idx, node_pos, target_idx, insert_pos, demand = best_move
                node = customer_routes[source_idx].pop(node_pos)
                customer_routes[target_idx].insert(insert_pos, node)
                loads[source_idx] -= demand
                loads[target_idx] += demand
                improved = True

            best_delta, best_swap = 0.0, None
            for first_idx in range(len(customer_routes)):
                for second_idx in range(first_idx + 1, len(customer_routes)):
                    first = customer_routes[first_idx]
                    second = customer_routes[second_idx]
                    old = (
                        self._route_distance(first, mdp)
                        + self._route_distance(second, mdp)
                    )
                    for i, first_node in enumerate(first):
                        for j, second_node in enumerate(second):
                            first_load = (
                                loads[first_idx]
                                - mdp.node_demand[first_node]
                                + mdp.node_demand[second_node]
                            )
                            second_load = (
                                loads[second_idx]
                                - mdp.node_demand[second_node]
                                + mdp.node_demand[first_node]
                            )
                            if (
                                first_load > mdp.cars_capacity + tolerance
                                or second_load > mdp.cars_capacity + tolerance
                            ):
                                continue
                            new_first, new_second = first.copy(), second.copy()
                            new_first[i], new_second[j] = second_node, first_node
                            delta = (
                                self._route_distance(new_first, mdp)
                                + self._route_distance(new_second, mdp)
                                - old
                            )
                            if delta < best_delta - tolerance:
                                best_delta = delta
                                best_swap = (
                                    first_idx,
                                    second_idx,
                                    i,
                                    j,
                                    first_load,
                                    second_load,
                                )
            if best_swap is not None:
                (
                    first_idx,
                    second_idx,
                    i,
                    j,
                    first_load,
                    second_load,
                ) = best_swap
                (
                    customer_routes[first_idx][i],
                    customer_routes[second_idx][j],
                ) = (
                    customer_routes[second_idx][j],
                    customer_routes[first_idx][i],
                )
                loads[first_idx], loads[second_idx] = first_load, second_load
                improved = True

        improved_routes = []
        for customers, load in zip(customer_routes, loads):
            path = [mdp.depot_num, *customers]
            if customers:
                path.append(mdp.depot_num)
            improved_routes.append(
                {
                    "path": path,
                    "total_distance": self._route_distance(customers, mdp),
                    "capacity": float(load),
                    "current_node": mdp.depot_num,
                }
            )
        return improved_routes

    def evaluate(self, mdp):
        _distance, routes = super().evaluate(mdp)
        routes = self._local_search(deepcopy(routes), mdp)
        self.routes = routes
        return sum(route["total_distance"] for route in routes), routes
