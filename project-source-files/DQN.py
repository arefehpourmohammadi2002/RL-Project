import copy
import itertools
import random

import numpy as np
import torch
import torch.nn as nn

from replay_buffer import ReplayBuffer
from MDP import MDP, apply_action
import GNN_embedding as GNN


class DQNetwork(nn.Module):
    def __init__(self, graph_embedding_size, node_embedding_size,
                 car_cap_size, total_dis_size, remaining_size, hidden_dim, output_dim):
        super().__init__()

        input_size = (graph_embedding_size
                      + node_embedding_size
                      + node_embedding_size
                      + car_cap_size
                      + total_dis_size
                      + remaining_size)

        self.dqn = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input):
        return self.dqn(input)


class DQN:

    def __init__(self, mdp, gnn_model, transformer_model, gnn_input, num_epoches,
                 explore_model_update_step, target_model_update_step,
                 epsilon, epsilon_decay, lr, hiden_dim, output_dim,
                 replay_buff_cap, replay_buffer_first_size, discount,
                 batch_size=8,
                 min_num_nodes=5, max_num_nodes=300,
                 min_num_cars=1, max_num_cars=100,
                 cars_capacity=120,
                 min_dis=1, max_dis=10,
                 min_node_cap=1, max_node_cap=60,
                 large_value=100, depot_num=0,
                 threshold_ema_alpha=0.3, max_stall_passes=20):

        self.mdp = mdp
        self.gnn_model = gnn_model
        self.transformer_model = transformer_model
        self.gnn_input = gnn_input
        self.num_epoches = num_epoches
        self.explore_model_update_step = explore_model_update_step
        self.target_model_update_step = target_model_update_step
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.batch_size = batch_size

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
        self.depot_num = depot_num

        self.threshold_ema_alpha = threshold_ema_alpha
        self.max_stall_passes = max_stall_passes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gnn_model.to(self.device)
        self.transformer_model.to(self.device)
        self.gnn_input = self.gnn_input.to(self.device)

        self.used = set()

        self.explore_update_counter = 0

        self.replay_buffer = ReplayBuffer(replay_buff_cap, min_num_nodes, max_num_nodes,
                                            min_num_cars, max_num_cars,
                                            cars_capacity,
                                            min_dis, max_dis,
                                            min_node_cap, max_node_cap,
                                            large_value)
        
        self.replay_buffer.full_buffer(replay_buffer_first_size)

        with torch.no_grad():
            self.graph_embedding, self.node_embedding = self.compute_embeddings(
                self.gnn_input, requires_grad=False
            )

        remaining_size = self.node_embedding.size(-1) + 1 # why 

        self.target_model = DQNetwork(self.graph_embedding.size(-1), self.node_embedding.size(-1),
                                       1, 1, remaining_size, hiden_dim, output_dim)
        self.explore_model = DQNetwork(self.graph_embedding.size(-1), self.node_embedding.size(-1),
                                        1, 1, remaining_size, hiden_dim, output_dim)
        self.target_model.load_state_dict(self.explore_model.state_dict())

        self.target_model.to(self.device)
        self.explore_model.to(self.device)

        self.optimizer = torch.optim.Adam(
            itertools.chain(
                self.explore_model.parameters(),
                self.gnn_model.parameters(),
                self.transformer_model.parameters()
            ),
            lr=lr
        )
        self.criterion = torch.nn.MSELoss()

        self.target_update_counter = 0

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

    def compute_embeddings(self, gnn_input, requires_grad=False):
        if requires_grad:
            node_embedding = self.gnn_model(gnn_input)
            graph_embedding = self.transformer_model(node_embedding)
        else:
            with torch.no_grad():
                node_embedding = self.gnn_model(gnn_input)
                graph_embedding = self.transformer_model(node_embedding)
        return graph_embedding, node_embedding

    def remaining_summary(self, node_embedding, mdp, used):

        unvisited = [i for i in range(mdp.num_nodes) if i not in used and i != mdp.depot_num]
        if not unvisited:
            mean_emb = torch.zeros(node_embedding.size(-1), device=self.device) # doesnt all zero cause env performace reduction
        else:
            idx = torch.tensor(unvisited, device=self.device)
            mean_emb = node_embedding[idx].mean(dim=0)
        frac = torch.tensor([len(unvisited) / max(1, mdp.num_nodes - 1)],
                             dtype=torch.float32, device=self.device)
        return torch.cat([mean_emb, frac], dim=-1)

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

    def evaluate_candidates(self, route, mdp, candidates, target=False,
                              graph_embedding=None, node_embedding=None, used=None):

        if target:
            model = self.target_model
        else:
            model = self.explore_model

        if graph_embedding is None:
            graph_embedding = self.graph_embedding
        if node_embedding is None:
            node_embedding = self.node_embedding
        if used is None:
            used = self.used

        remaining = self.remaining_summary(node_embedding, mdp, used)

        candidates_t = torch.tensor(candidates, device=self.device, dtype=torch.long)
        num_candidates = candidates_t.size(0)

        cap_tensor = torch.tensor([route["capacity"]], dtype=torch.float32, device=self.device)
        dis_tensor = torch.tensor([route["total_distance"]], dtype=torch.float32, device=self.device)

        graph_rep = graph_embedding.unsqueeze(0).expand(num_candidates, -1)
        current_rep = node_embedding[route["current_node"]].unsqueeze(0).expand(num_candidates, -1)
        candidate_emb = node_embedding[candidates_t]
        cap_rep = cap_tensor.unsqueeze(0).expand(num_candidates, -1)
        dis_rep = dis_tensor.unsqueeze(0).expand(num_candidates, -1)
        remaining_rep = remaining.unsqueeze(0).expand(num_candidates, -1)

        batch_input = torch.cat([
            graph_rep, current_rep, candidate_emb, cap_rep, dis_rep, remaining_rep
        ], dim=-1)

        with torch.no_grad():
            return model(batch_input).squeeze(-1)

    def find_best_action(self, route, mdp, target=False, graph_embedding=None,
                          node_embedding=None, used=None):

        candidates = self.get_candidates(route, mdp, used)
        if not candidates:
            return float('-inf'), None

        q_values = self.evaluate_candidates(
            route, mdp, candidates, target=target,
            graph_embedding=graph_embedding, node_embedding=node_embedding, used=used
        )
        best_idx = torch.argmax(q_values).item()
        return q_values[best_idx].item(), candidates[best_idx]

    def e_greedy_policy(self, route, mdp, candidates):
 
        q_values = self.evaluate_candidates(route, mdp, candidates, target=False)
        best_idx = torch.argmax(q_values).item()
        q_max = q_values[best_idx].item()

        if random.random() < self.epsilon:
            exec_idx = random.randrange(len(candidates))
        else:
            exec_idx = best_idx

        return q_max, candidates[exec_idx]

    def execute(self, routes, mdp, training, k, node, q_max):
        route = routes[k]
        pre_state = copy.deepcopy(route)
        distance = mdp.distance_matrix[route["current_node"], node]
        reward = -1.0 * distance
        route["path"].append(node)
        route["capacity"] += mdp.node_capacity[node]
        route["total_distance"] += distance
        route["current_node"] = node
        self.used.add(node)

        if training:
            self.replay_buffer.insert(pre_state, node, reward, self.used, mdp, self.gnn_input)
            self.explore_update_counter += 1
            if (self.explore_update_counter >= self.explore_model_update_step
                    and len(self.replay_buffer.buffer) >= self.batch_size):
                self.train_step()
                self.explore_update_counter = 0
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def run_episode(self, mdp, gnn_input, training):
        self.mdp = mdp
        self.gnn_input = gnn_input.to(self.device)

        routes = self.build_routes(mdp)
        with torch.no_grad():
            self.graph_embedding, self.node_embedding = self.compute_embeddings(
                self.gnn_input, requires_grad=False
            )

        finished = {k: False for k in routes}
        pool_ema = None
        stall_passes = 0

        while not all(finished.values()):
            proposals = {}
            terminal_now = []

            for k, route in routes.items():
                if finished[k]:
                    continue
                if route["capacity"] >= mdp.cars_capacity:
                    finished[k] = True
                    continue

                candidates = self.get_candidates(route, mdp)
                if not candidates:
                    terminal_now.append(k)
                    continue

                proposals[k] = self.e_greedy_policy(route, mdp, candidates, training)

            for k in terminal_now:
                route = routes[k]
                pre_state = copy.deepcopy(route)
                distance = mdp.distance_matrix[route["current_node"]][mdp.depot_num]
                reward = -1.0 * distance
                route["total_distance"] += distance
                finished[k] = True
                if training:
                    self.replay_buffer.insert(pre_state, None, reward, self.used, mdp, self.gnn_input)

            accepted_keys = []
            for k, (q_max, node) in proposals.items():
                route_len = len(routes[k]["path"])
                threshold = float('-inf') if pool_ema is None else pool_ema / route_len
                if q_max >= threshold:
                    accepted_keys.append(k)

            executed_keys = []
            accepted_keys.sort(key=lambda kk: proposals[kk][0], reverse=True)
            for k in accepted_keys:
                q_max, node = proposals[k]
                if node in self.used:
                    continue
                self.execute(routes, mdp, training, k, node, q_max)
                executed_keys.append(k)

            if proposals:
                pass_mean = sum(q for q, _ in proposals.values()) / len(proposals)
                pool_ema = pass_mean if pool_ema is None else (
                    self.threshold_ema_alpha * pass_mean
                    + (1 - self.threshold_ema_alpha) * pool_ema
                )

            if executed_keys or terminal_now:
                stall_passes = 0
            elif proposals:
                stall_passes += 1

            if stall_passes >= self.max_stall_passes and proposals:
                k = max(proposals, key=lambda kk: proposals[kk][0])
                q_max, node = proposals[k]
                self.execute(routes, mdp, training, k, node, q_max)
                stall_passes = 0

            if len(self.used) >= mdp.num_nodes - 1:
                break

        for k, route in routes.items():
            if route["current_node"] != mdp.depot_num:
                route["path"].append(mdp.depot_num)
                route["total_distance"] += mdp.distance_matrix[route["current_node"]][mdp.depot_num]

        return routes

    def embeddings_for(self, embeddings_cache, mdp, gnn_input):
        key = id(mdp)
        if key not in embeddings_cache:
            embeddings_cache[key] = self.compute_embeddings(gnn_input, requires_grad=True)
        return embeddings_cache[key]

    def train_step(self):
        routes, actions, rewards, used_sets, mdps, gnn_inputs = self.replay_buffer.sample(self.batch_size)

        embeddings_cache = {}

        current_q_list = []
        target_q_list = []

        for route, action, reward, used, mdp, gnn_input in zip(
                routes, actions, rewards, used_sets, mdps, gnn_inputs):
            if action is None:
                continue

            graph_embedding, node_embedding = self.embeddings_for(embeddings_cache, mdp, gnn_input)

            remaining = self.remaining_summary(node_embedding, mdp, used)

            cap_tensor = torch.tensor([route["capacity"]], dtype=torch.float32, device=self.device)
            dis_tensor = torch.tensor([route["total_distance"]], dtype=torch.float32, device=self.device)

            inp = torch.cat([
                graph_embedding,
                node_embedding[route["current_node"]],
                node_embedding[action],
                cap_tensor,
                dis_tensor,
                remaining
            ], dim=-1)

            current_q_list.append(self.explore_model(inp))

            next_route = apply_action(route, action, mdp)
            with torch.no_grad():
                q_max = 0.0
                if next_route["capacity"] < mdp.cars_capacity:
                    next_used = used | {action}
                    q_max, best_action = self.find_best_action(
                        next_route, mdp, target=True,
                        graph_embedding=graph_embedding.detach(),
                        node_embedding=node_embedding.detach(),
                        used=next_used
                    )
                    if best_action is None:
                        q_max = 0.0
                y_value = reward + self.discount * q_max
            target_q_list.append(torch.tensor([y_value], dtype=torch.float32, device=self.device))

        if not current_q_list:
            return

        current_q = torch.cat(current_q_list)
        target_q = torch.cat(target_q_list).detach()

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_update_counter += 1
        if self.target_update_counter >= self.target_model_update_step:
            self.target_model.load_state_dict(self.explore_model.state_dict())
            self.target_update_counter = 0

        with torch.no_grad():
            self.graph_embedding, self.node_embedding = self.compute_embeddings(
                self.gnn_input, requires_grad=False
            )

    def DQN_train(self):
        for i in range(self.num_epoches):
            if i == 0:
                mdp, gnn_input = self.mdp, self.gnn_input
            else:
                mdp, gnn_input = self.generate_random_env()
            self.run_episode(mdp, gnn_input, training=True)

    def eval_model(self, mdp=None, gnn_input=None):
        if mdp is None or gnn_input is None:
            mdp, gnn_input = self.generate_random_env()
        return self.run_episode(mdp, gnn_input, training=False)