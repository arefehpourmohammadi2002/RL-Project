import copy
import itertools
import random

import torch
import torch.nn as nn

from replay_buffer import ReplayBuffer
from MDP import apply_action


class DQNetwork(nn.Module):
    def __init__(self, graph_embedding_size, node_embedding_size,
                 car_cap_size, total_dis_size, hidden_dim, output_dim):
        super().__init__()

        input_size = (graph_embedding_size
                      + node_embedding_size
                      + node_embedding_size
                      + car_cap_size
                      + total_dis_size)

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
                 batch_size=8):

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_buffer = ReplayBuffer(replay_buff_cap)
        self.replay_buffer.full_buffer(replay_buffer_first_size, mdp)
        self.used = self.replay_buffer.used

        
        with torch.no_grad():
            self.graph_embedding, self.node_embedding = self.compute_embeddings(requires_grad=False)

        self.target_model = DQNetwork(self.graph_embedding.size(-1), self.node_embedding.size(-1),
                                       1, 1, hiden_dim, output_dim)
        self.explore_model = DQNetwork(self.graph_embedding.size(-1), self.node_embedding.size(-1),
                                        1, 1, hiden_dim, output_dim)
        self.target_model.load_state_dict(self.explore_model.state_dict())

        self.gnn_model.to(self.device)
        self.transformer_model.to(self.device)
        self.target_model.to(self.device)
        self.explore_model.to(self.device)
        self.gnn_input = self.gnn_input.to(self.device)

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

    def refresh_env(self):

        routes = {
            f"route{k}": {
                "path": [self.mdp.depot_num],
                "capacity": 0.0,
                "total_distance": 0.0,
                "current_node": self.mdp.depot_num
            }
            for k in range(0, self.mdp.num_cars)
        }
        return routes

    def compute_embeddings(self, requires_grad=False):
        if requires_grad:
            node_embedding = self.gnn_model(self.gnn_input)
            graph_embedding = self.transformer_model(node_embedding)
        else:
            with torch.no_grad():
                node_embedding = self.gnn_model(self.gnn_input)
                graph_embedding = self.transformer_model(node_embedding)
        return graph_embedding, node_embedding

    def find_best_action(self, route, target=False, graph_embedding=None, node_embedding=None):
        if target:
            model = self.target_model 
        else:
            model = self.explore_model

        if graph_embedding is None:
            graph_embedding = self.graph_embedding 
        if node_embedding is None:
            node_embedding = self.node_embedding 

        q_max = float('-inf')
        max_action = None
        with torch.no_grad():
            for i in range(self.mdp.num_nodes):
                if i in self.used or i == route["current_node"]:
                    continue
                if route["capacity"] + self.mdp.node_capacity[i] > self.mdp.cars_capacity:
                    continue

                cap_tensor = torch.tensor([route["capacity"]], dtype=torch.float32, device=self.device)
                dis_tensor = torch.tensor([route["total_distance"]], dtype=torch.float32, device=self.device)

                input = torch.cat([
                    graph_embedding,
                    node_embedding[route["current_node"]],
                    node_embedding[i],
                    cap_tensor,
                    dis_tensor
                ], dim=-1)

                q_prim = model(input).item()

                if q_prim > q_max:
                    q_max = q_prim
                    max_action = i

        return q_max, max_action

    def e_greedy_policy(self, route):

        if random.random() < self.epsilon:
            candidates = [
                i for i in range(self.mdp.num_nodes)
                if i not in self.used
                and i != route["current_node"]
                and route["capacity"] + self.mdp.node_capacity[i] <= self.mdp.cars_capacity
            ]
            if not candidates:
                return None, None
            return None, random.choice(candidates)

        return self.find_best_action(route)

    def train_step(self):
        routes, actions, rewards = self.replay_buffer.sample(self.batch_size)
        graph_embedding, node_embedding = self.compute_embeddings(requires_grad=True)

        current_q_list = []
        target_q_list = []

        for route, action, reward in zip(routes, actions, rewards):
            cap_tensor = torch.tensor([route["capacity"]], dtype=torch.float32, device=self.device)
            dis_tensor = torch.tensor([route["total_distance"]], dtype=torch.float32, device=self.device)

            inp = torch.cat([
                graph_embedding,
                node_embedding[route["current_node"]],
                node_embedding[action],
                cap_tensor,
                dis_tensor
            ], dim=-1)

            current_q_list.append(self.explore_model(inp))

            next_route = self.mdp.apply_action(route, action)
            with torch.no_grad():
                q_max = 0.0
                if next_route["capacity"] < self.mdp.cars_capacity:
                    q_max, best_action = self.find_best_action(
                        next_route, target=True,
                        graph_embedding=graph_embedding.detach(),
                        node_embedding=node_embedding.detach()
                    )
                    if best_action is None:
                        q_max = 0.0
                y_value = reward + self.discount * q_max
            target_q_list.append(torch.tensor([y_value], dtype=torch.float32, device=self.device))

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
            self.graph_embedding, self.node_embedding = self.compute_embeddings(requires_grad=False)


    def DQN_train(self):
        explore_update = 0
        
        for _ in range(self.num_epoches):
            self.routes = self.refresh_env()
            with torch.no_grad():
                self.graph_embedding, self.node_embedding = self.compute_embeddings(requires_grad=False)

            for _, route in self.routes.items():
                if route["capacity"] >= self.mdp.cars_capacity:
                    continue

                pre_state = copy.deepcopy(route)
                _, next_state = self.e_greedy_policy(route)
                if next_state is None:
                    continue

                if route["capacity"] + self.mdp.node_capacity[next_state] > self.mdp.cars_capacity:
                    reward = -1000.0
                else:
                    distance = self.mdp.distance_matrix[route["current_node"], next_state]
                    reward = -1.0 * distance
                    route["path"].append(next_state)
                    route["capacity"] += self.mdp.node_capacity[next_state] 
                    route["total_distance"] += distance
                    route["current_node"] = next_state
                    self.used.add(next_state)

                self.replay_buffer.insert(pre_state, next_state, reward)

                explore_update += 1
                if (explore_update >= self.explore_model_update_step
                        and len(self.replay_buffer.buffer) >= self.batch_size):
                    self.train_step()
                    explore_update = 0
                    
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        # for name, param in self.explore_model.named_parameters():
        #     print(name, param.shape, param)

    def eval_model(self):
        eval_routs = self.refresh_env()
        self.used.clear()
        finished = {k: False for k in eval_routs}

        while True:
            for k, route in eval_routs.items():
                if finished[k]:
                    continue

                if route["capacity"] >= self.mdp.cars_capacity:
                    finished[k] = True
                    continue

                _, node = self.find_best_action(route)

                if node is not None:
                    distance = self.mdp.distance_matrix[route["current_node"], node]
                    self.used.add(node)
                    route["path"].append(node)
                    route["capacity"] += self.mdp.node_capacity[node]
                    route["total_distance"] += distance
                    route["current_node"] = node
                else:
                    
                    route["total_distance"] += self.mdp.distance_matrix[route["current_node"]][self.mdp.depot_num]
                    finished[k] = True

            if len(self.used) == self.mdp.num_nodes - 1 or all(finished.values()):
                return eval_routs