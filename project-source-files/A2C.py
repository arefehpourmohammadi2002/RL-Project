import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CNetwork(nn.Module):
    def __init__(self, graph_embedding_size, node_embedding_size,
                 car_cap_size, total_dis_size, hidden_dim, output_dim):
        super().__init__()

        input_size_critic = (graph_embedding_size
                              + node_embedding_size
                              + car_cap_size
                              + total_dis_size)

        self.value_function = nn.Sequential(
            nn.Linear(input_size_critic, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        input_size_action = (graph_embedding_size
                              + node_embedding_size
                              + node_embedding_size
                              + car_cap_size
                              + total_dis_size)


        self.policy_head = nn.Sequential(
            nn.Linear(input_size_action, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input, critic=True):
        if critic:
            return self.value_function(input)
        return self.policy_head(input)


class A2C:
    def __init__(self, mdp, gnn_model, transformer_model, gnn_input,
                 num_epoches, lr, hidden_dim, output_dim, discount,
                 entropy_coef=0.01):

        self.mdp = mdp
        self.gnn_model = gnn_model
        self.transformer_model = transformer_model
        self.gnn_input = gnn_input
        self.num_epoches = num_epoches
        self.discount = discount
        self.entropy_coef = entropy_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.used = set()

        self.gnn_model.to(self.device)
        self.transformer_model.to(self.device)
        self.gnn_input = self.gnn_input.to(self.device)

        with torch.no_grad():
            self.graph_embedding, self.node_embedding = self.compute_embeddings(requires_grad=False)

        self.A2C_network = A2CNetwork(self.graph_embedding.size(-1), self.node_embedding.size(-1),
                                       1, 1, hidden_dim, output_dim)
        self.A2C_network.to(self.device)


        self.optimizer = torch.optim.Adam(
            itertools.chain(
                self.A2C_network.parameters(),
                self.gnn_model.parameters(),
                self.transformer_model.parameters()
            ),
            lr=lr
        )

        self.critic_criterion = torch.nn.MSELoss()

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

    def policy_probabilities(self, route):

        cap_tensor = torch.tensor([route["capacity"]], dtype=torch.float32, device=self.device)
        dis_tensor = torch.tensor([route["total_distance"]], dtype=torch.float32, device=self.device)

        logits = []
        actions = []
        for i in range(self.mdp.num_nodes):
            if i in self.used or i == route["current_node"]:
                continue
            if route["capacity"] + self.mdp.node_capacity[i] > self.mdp.cars_capacity:
                continue

            input_policy = torch.cat([
                self.graph_embedding,
                self.node_embedding[route["current_node"]],
                self.node_embedding[i],
                cap_tensor,
                dis_tensor
            ], dim=-1)

            logit = self.A2C_network(input_policy, critic=False)  
            logits.append(logit)
            actions.append(i)

        if not actions:
            return None, None

        logits = torch.cat(logits)       
        probs = F.softmax(logits, dim=-1)
        return probs, actions

    def select_action(self, route):
        probs, actions = self.policy_probabilities(route)
        if actions is None:
            return None, None, None

        with torch.no_grad():
            idx = torch.multinomial(probs, num_samples=1).item()

        chosen_action = actions[idx]
        log_prob = torch.log(probs[idx] + 1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()

        return chosen_action, log_prob, entropy

    def value_function(self, route):
        cap_tensor = torch.tensor([route["capacity"]], dtype=torch.float32, device=self.device)
        dis_tensor = torch.tensor([route["total_distance"]], dtype=torch.float32, device=self.device)

        input_val_fun = torch.cat([
            self.graph_embedding,
            self.node_embedding[route["current_node"]],
            cap_tensor,
            dis_tensor
        ], dim=-1)

        return self.A2C_network(input_val_fun, critic=True)

    def A2C_train(self):

        for _ in range(self.num_epoches):
            self.routes = self.refresh_env()
            self.used = set()
            finished = {k: False for k in self.routes}

            self.graph_embedding, self.node_embedding = self.compute_embeddings(requires_grad=True)

            while not all(finished.values()):
                progressed = False

                for k, route in self.routes.items():
                    if finished[k]:
                        continue
                    if route["capacity"] >= self.mdp.cars_capacity:
                        finished[k] = True
                        continue

                    current_value = self.value_function(route)

                    action, log_prob, entropy = self.select_action(route)

                    if action is None:

                        distance = self.mdp.distance_matrix[route["current_node"]][self.mdp.depot_num]
                        reward = -1.0 * distance
                        route["total_distance"] += distance
                        finished[k] = True

                        target = torch.tensor([reward], dtype=torch.float32, device=self.device)
                        loss = self.critic_criterion(current_value, target)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    else:
                        distance = self.mdp.distance_matrix[route["current_node"], action]
                        reward = -1.0 * distance


                        route["path"].append(action)
                        route["capacity"] += self.mdp.node_capacity[action]
                        route["total_distance"] += distance
                        route["current_node"] = action
                        self.used.add(action)  

                        done = (route["capacity"] >= self.mdp.cars_capacity
                                or len(self.used) >= self.mdp.num_nodes - 1)

                        with torch.no_grad():
                            next_value = (torch.zeros(1, device=self.device) if done
                                          else self.value_function(route))
                        target = reward + self.discount * next_value
                        advantage = (target - current_value).detach()

                        critic_loss = self.critic_criterion(current_value, target)

                        actor_loss = -log_prob * advantage
                        entropy_bonus = self.entropy_coef * entropy
                        loss = critic_loss + actor_loss - entropy_bonus

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        if done:
                            finished[k] = True

                    progressed = True


                    self.graph_embedding, self.node_embedding = self.compute_embeddings(requires_grad=True)

                if len(self.used) >= self.mdp.num_nodes - 1 or not progressed:
                    break

    def eval_model(self):
        eval_routes = self.refresh_env()
        self.used = set()
        finished = {k: False for k in eval_routes}

        with torch.no_grad():
            self.graph_embedding, self.node_embedding = self.compute_embeddings(requires_grad=False)

            while True:
                for k, route in eval_routes.items():
                    if finished[k]:
                        continue
                    if route["capacity"] >= self.mdp.cars_capacity:
                        finished[k] = True
                        continue

                    probs, actions = self.policy_probabilities(route)
                    if actions is None:
                        finished[k] = True
                        continue

                    best_idx = torch.argmax(probs).item()
                    node = actions[best_idx]

                    distance = self.mdp.distance_matrix[route["current_node"], node]
                    self.used.add(node)
                    route["path"].append(node)
                    route["capacity"] += self.mdp.node_capacity[node]
                    route["total_distance"] += distance
                    route["current_node"] = node

                if len(self.used) == self.mdp.num_nodes - 1 or all(finished.values()):
                    for k, route in eval_routes.items():
                        if route["current_node"] != self.mdp.depot_num:
                            route["path"].append(self.mdp.depot_num)
                            route["total_distance"] += self.mdp.distance_matrix[route["current_node"]][self.mdp.depot_num]
                    return eval_routes