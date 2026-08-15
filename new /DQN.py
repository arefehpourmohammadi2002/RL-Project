import copy
import random

import numpy as np
import torch
import torch.nn as nn

from replay_buffer import ReplayBuffer
from MDP import MDP, apply_action


class DQNetwork(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim):
        super().__init__()
        self.dqn = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input):
        return self.dqn(input)


class DQN:

    NODE_FEATURE_DIM = 4

    GRAPH_SUMMARY_DIM = 5

    def __init__(self, mdp, num_epoches,
                 explore_model_update_step, target_model_update_step,
                 epsilon, epsilon_decay, lr, hiden_dim, output_dim,
                 replay_buff_cap, replay_buffer_first_size, discount,
                 batch_size=8,
                 min_num_nodes=5, max_num_nodes=300,
                 min_num_cars=1, max_num_cars=100,
                 cars_capacity=120,
                 min_dis=1, max_dis=10,
                 min_node_cap=1, max_node_cap=60,
                 depot_num=0,
                 lr_decay=1.0, min_lr=1e-6, max_grad_norm=1000.0):
        # BUGFIX / REDESIGN: this used to run every node/graph through a GNN
        # + graph transformer to build learned embeddings. Confirmed across
        # many tests (see conversation) that pipeline collapses different
        # graphs toward near-identical outputs -- tried Pre-LN, input
        # normalization, sum vs mean aggregation, and rescaling the self-loop
        # placeholder; none fixed it, and the raw problem instances
        # themselves already share substantial baseline similarity (~0.6-0.9
        # cosine similarity) just from being drawn out of the same narrow
        # random ranges, so there wasn't much headroom for a learned
        # embedding to add on top anyway. Replaced entirely with raw features
        # computed directly from the MDP (node_features/graph_summary below)
        # -- guaranteed to differ meaningfully across instances, since
        # nothing here passes through any part of the pipeline that was
        # collapsing.

        self.mdp = mdp
        self.num_epoches = num_epoches
        self.explore_model_update_step = explore_model_update_step
        self.target_model_update_step = target_model_update_step
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.batch_size = batch_size

        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.max_grad_norm = max_grad_norm

        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.min_num_cars = min_num_cars
        self.max_num_cars = max_num_cars
        self.cars_capacity = cars_capacity
        self.min_dis = min_dis
        self.max_dis = max_dis
        self.min_node_cap = min_node_cap
        self.max_node_cap = max_node_cap
        self.depot_num = depot_num

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.used = set()
        self.explore_update_counter = 0

        self.replay_buffer = ReplayBuffer(replay_buff_cap, min_num_nodes, max_num_nodes,
                                            min_num_cars, max_num_cars,
                                            cars_capacity,
                                            min_dis, max_dis,
                                            min_node_cap, max_node_cap,
                                            self.device,
                                            self.depot_num)

        self.replay_buffer.full_buffer(replay_buffer_first_size)

        # Q-network input layout, all raw / directly computed from the MDP:
        #   graph_summary              : GRAPH_SUMMARY_DIM
        #   node_features[current]     : NODE_FEATURE_DIM
        #   node_features[candidate]   : NODE_FEATURE_DIM
        #   distance(current, candidate) : 1
        #   route capacity             : 1
        #   route total_distance       : 1
        #   remaining summary (mean node_features of unvisited + frac) : NODE_FEATURE_DIM + 1
        input_size = (self.GRAPH_SUMMARY_DIM
                      + self.NODE_FEATURE_DIM * 2
                      + 1 + 1 + 1
                      + self.NODE_FEATURE_DIM + 1)

        self.target_model = DQNetwork(input_size, hiden_dim, output_dim)
        self.explore_model = DQNetwork(input_size, hiden_dim, output_dim)
        self.target_model.load_state_dict(self.explore_model.state_dict())

        self.target_model.to(self.device)
        self.explore_model.to(self.device)

        self.optimizer = torch.optim.Adam(self.explore_model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.target_update_counter = 0

    def generate_random_env(self):
        num_nodes = np.random.randint(self.min_num_nodes, self.max_num_nodes + 1)
        num_cars = np.random.randint(self.min_num_cars, self.max_num_cars + 1)
        cars_capacity = np.random.uniform(1, self.cars_capacity)

        mdp = MDP(num_nodes, self.depot_num, num_cars, cars_capacity)
        mdp.fill_distance_matrix(self.min_dis, self.max_dis)
        mdp.fill_node_cap_matrix(self.min_node_cap, self.max_node_cap)

        return mdp

    def build_routes(self, mdp):
        self.used.clear()
        return {
            f"route{k}": {
                "path": [mdp.depot_num],
                "capacity": 0.0,
                "total_distance": 0.0,
                "current_node": mdp.depot_num
            }
            for k in range(0, mdp.num_cars)
        }

    def node_features(self, mdp):
        node_capacity = np.asarray(mdp.node_capacity, dtype=np.float64)
        dm = np.asarray(mdp.distance_matrix, dtype=np.float64)

        dist_from_depot = dm[mdp.depot_num, :].copy()
        dist_from_depot[mdp.depot_num] = 0.0  # depot-to-itself is inf, not meaningful

        finite_mask = np.isfinite(dm)
        mean_dist = np.zeros(mdp.num_nodes)
        min_dist = np.zeros(mdp.num_nodes)
        for i in range(mdp.num_nodes):
            row = dm[i][finite_mask[i]]
            if row.size > 0:
                mean_dist[i] = row.mean()
                min_dist[i] = row.min()

        feats = np.stack([node_capacity, dist_from_depot, mean_dist, min_dist], axis=1)
        feat_mean = feats.mean(axis=0, keepdims=True)
        feat_std = feats.std(axis=0, keepdims=True) + 1e-6
        feats = (feats - feat_mean) / feat_std

        return torch.tensor(feats, dtype=torch.float32, device=self.device)

    def graph_summary(self, mdp):
        """Explicit graph-level statistics, computed directly from the MDP
        instance. See node_features() -- same rationale."""
        off_diag = ~np.eye(mdp.num_nodes, dtype=bool)
        distances = mdp.distance_matrix[off_diag]

        total_demand = float(mdp.node_capacity.sum())

        stats = [
            float(mdp.num_nodes),
            float(mdp.num_cars),
            float(distances.mean()),
            float(distances.std()),
            total_demand / mdp.cars_capacity,
        ]
        return torch.tensor(stats, dtype=torch.float32, device=self.device)

    def remaining_summary(self, node_feats, mdp, used):
        unvisited = [i for i in range(mdp.num_nodes) if i not in used and i != mdp.depot_num]
        if not unvisited:
            mean_feat = torch.zeros(node_feats.size(-1), device=self.device)
        else:
            idx = torch.tensor(unvisited, device=self.device)
            mean_feat = node_feats[idx].mean(dim=0)
        frac = torch.tensor([len(unvisited) / max(1, mdp.num_nodes - 1)],
                             dtype=torch.float32, device=self.device)
        return torch.cat([mean_feat, frac], dim=-1)

    def get_candidates(self, route, mdp, used=None):
        if used is None:
            used = self.used

        node_capacity = mdp.node_capacity
        remaining_cap = mdp.cars_capacity - route["capacity"]

        mask = node_capacity <= remaining_cap
        mask[route["current_node"]] = False
        mask[mdp.depot_num] = False
        if used:
            used_idx = np.fromiter(used, dtype=int)
            mask[used_idx] = False

        return np.nonzero(mask)[0].tolist()

    def _batch_rows(self, routes, mdp, candidates_by_route, node_feats, graph_sum, remaining_row):
        """Build (all_pairs, batch_input) for a set of {route_key:
        candidates} -- shared by select_global_action and
        compute_global_max_q."""
        all_pairs = []
        rows = []

        for k, candidates in candidates_by_route.items():
            route = routes[k]
            current_feat = node_feats[route["current_node"]]
            cap_val = route["capacity"]
            dis_val = route["total_distance"]

            for node in candidates:
                dist_val = float(mdp.distance_matrix[route["current_node"], node])
                row = torch.cat([
                    graph_sum,
                    current_feat,
                    node_feats[node],
                    torch.tensor([dist_val], dtype=torch.float32, device=self.device),
                    torch.tensor([cap_val], dtype=torch.float32, device=self.device),
                    torch.tensor([dis_val], dtype=torch.float32, device=self.device),
                    remaining_row,
                ], dim=-1)
                rows.append(row)
                all_pairs.append((k, node))

        if not rows:
            return all_pairs, None
        return all_pairs, torch.stack(rows)

    def select_global_action(self, routes, mdp, active_candidates, training):
        node_feats = self.node_features(mdp)
        graph_sum = self.graph_summary(mdp)
        remaining_row = self.remaining_summary(node_feats, mdp, self.used)

        all_pairs, batch_input = self._batch_rows(
            routes, mdp, active_candidates, node_feats, graph_sum, remaining_row
        )

        with torch.no_grad():
            all_q = self.explore_model(batch_input).squeeze(-1)

        best_idx = torch.argmax(all_q).item()

        if training and random.random() < self.epsilon:
            exec_idx = random.randrange(len(all_pairs))
        else:
            exec_idx = best_idx

        k, node = all_pairs[exec_idx]
        return all_q[best_idx].item(), k, node

    def compute_global_max_q(self, routes, mdp, used, node_feats, graph_sum):
        """Read-only Double DQN bootstrap value: explore_model selects the
        best action, target_model evaluates it. See conversation for why
        (plain target_model.max() lets the same noisy network both nominate
        and confirm its own optimistic estimate)."""
        candidates_by_route = {}
        for k, route in routes.items():
            if route["capacity"] >= mdp.cars_capacity:
                continue
            candidates = self.get_candidates(route, mdp, used)
            if not candidates:
                continue
            candidates_by_route[k] = candidates

        if not candidates_by_route:
            return 0.0

        remaining_row = self.remaining_summary(node_feats, mdp, used)
        all_pairs, batch_input = self._batch_rows(
            routes, mdp, candidates_by_route, node_feats, graph_sum, remaining_row
        )

        with torch.no_grad():
            explore_q = self.explore_model(batch_input).squeeze(-1)
            best_idx = torch.argmax(explore_q).item()
            target_q = self.target_model(batch_input).squeeze(-1)
            return target_q[best_idx].item()

    def execute(self, routes, mdp, training, k, node, q_max):
        route = routes[k]
        routes_snapshot = copy.deepcopy(routes)
        used_snapshot = self.used.copy()
        distance = mdp.distance_matrix[route["current_node"], node]
        reward = -1.0 * distance
        route["path"].append(node)
        route["capacity"] += mdp.node_capacity[node]
        route["total_distance"] += distance
        route["current_node"] = node
        self.used.add(node)

        if training:
            self.replay_buffer.insert(routes_snapshot, k, node, reward, used_snapshot, mdp)
            self.explore_update_counter += 1
            if (self.explore_update_counter >= self.explore_model_update_step
                    and len(self.replay_buffer.buffer) >= self.batch_size):
                self.train_step()
                self.explore_update_counter = 0
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def run_episode(self, mdp, training):
        self.mdp = mdp
        routes = self.build_routes(mdp)
        finished = {k: False for k in routes}

        while not all(finished.values()):
            active_candidates = {}

            for k, route in routes.items():
                if finished[k]:
                    continue

                if route["capacity"] >= mdp.cars_capacity:
                    finished[k] = True
                    if route["current_node"] != mdp.depot_num:
                        pre_state = copy.deepcopy(route)
                        used_snapshot = self.used.copy()
                        distance = mdp.distance_matrix[route["current_node"]][mdp.depot_num]
                        reward = -1.0 * distance
                        route["total_distance"] += distance
                        route["path"].append(mdp.depot_num)
                        route["current_node"] = mdp.depot_num
                        if training:
                            self.replay_buffer.insert({k: pre_state}, k, None, reward, used_snapshot, mdp)
                    continue

                candidates = self.get_candidates(route, mdp)
                if not candidates:
                    finished[k] = True
                    if route["current_node"] != mdp.depot_num:
                        pre_state = copy.deepcopy(route)
                        used_snapshot = self.used.copy()
                        distance = mdp.distance_matrix[route["current_node"]][mdp.depot_num]
                        reward = -1.0 * distance
                        route["total_distance"] += distance
                        route["path"].append(mdp.depot_num)
                        route["current_node"] = mdp.depot_num
                        if training:
                            self.replay_buffer.insert({k: pre_state}, k, None, reward, used_snapshot, mdp)
                    continue

                active_candidates[k] = candidates

            if not active_candidates:
                break

            q_max, k, node = self.select_global_action(routes, mdp, active_candidates, training)
            self.execute(routes, mdp, training, k, node, q_max)

        return routes

    def train_step(self):
        routes_snapshots, acting_keys, actions, rewards, used_sets, mdps = self.replay_buffer.sample(self.batch_size)

        node_feats_cache = {}

        def get_node_feats(mdp):
            key = id(mdp)
            if key not in node_feats_cache:
                node_feats_cache[key] = self.node_features(mdp)
            return node_feats_cache[key]

        current_q_inputs = []
        target_q_list = []

        for routes_snapshot, acting_key, action, reward, used, mdp in zip(
                routes_snapshots, acting_keys, actions, rewards, used_sets, mdps):

            node_feats = get_node_feats(mdp)
            graph_sum = self.graph_summary(mdp)
            route = routes_snapshot[acting_key]
            remaining = self.remaining_summary(node_feats, mdp, used)

            action_node = action if action is not None else mdp.depot_num
            if action is not None:
                dist_val = float(mdp.distance_matrix[route["current_node"], action_node])
            else:
                # Closing transition: there's no real "candidate" (the route
                # is ending, not moving to a node), so there's no meaningful
                # current->candidate distance to report -- 0 is a neutral
                # placeholder.
                dist_val = 0.0

            current_q_inputs.append(torch.cat([
                graph_sum,
                node_feats[route["current_node"]],
                node_feats[action_node],
                torch.tensor([dist_val], dtype=torch.float32, device=self.device),
                torch.tensor([route["capacity"]], dtype=torch.float32, device=self.device),
                torch.tensor([route["total_distance"]], dtype=torch.float32, device=self.device),
                remaining
            ], dim=-1))

            if action is None:
                y_value = reward
            else:
                next_routes = dict(routes_snapshot)
                next_routes[acting_key] = apply_action(route, action, mdp)
                next_used = used | {action}

                q_max = self.compute_global_max_q(next_routes, mdp, next_used, node_feats, graph_sum)
                y_value = reward + self.discount * q_max
            target_q_list.append(torch.tensor([y_value], dtype=torch.float32, device=self.device))

        if not current_q_inputs:
            return

        current_q = self.explore_model(torch.stack(current_q_inputs)).squeeze(-1)
        target_q = torch.cat(target_q_list).detach()

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.explore_model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = max(self.min_lr, param_group["lr"] * self.lr_decay)

        self.target_update_counter += 1
        if self.target_update_counter >= self.target_model_update_step:
            self.target_model.load_state_dict(self.explore_model.state_dict())
            self.target_update_counter = 0

    def DQN_train(self):
        for i in range(self.num_epoches):
            if i == 0:
                mdp = self.mdp
            else:
                mdp = self.generate_random_env()
            self.run_episode(mdp, training=True)
            print(i)
        if self.lr_decay != 1.0:
            final_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Final learning rate after decay: {final_lr:.6g}")

    def eval_model(self, mdp=None):
        if mdp is None:
            mdp = self.generate_random_env()
        return self.run_episode(mdp, training=False)
