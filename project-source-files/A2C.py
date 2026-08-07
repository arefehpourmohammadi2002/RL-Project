import copy
import itertools
import random

import torch
import torch.nn as nn

from replay_buffer import ReplayBuffer
from MDP import apply_action

def create_train_env():
    pass

class A2CNetwork(nn.Module):
    def __init__(self, graph_embedding_size, node_embedding_size,
                 car_cap_size, total_dis_size, hidden_dim, output_dim):
        super().__init__()

        input_size_critic = (graph_embedding_size
                      + node_embedding_size
                      + car_cap_size
                      + total_dis_size)

        self.value_functon = nn.Sequential(
            nn.Linear(input_size_critic, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        input_size_action = (graph_embedding_size
                      + node_embedding_size
                      + node_embedding_size
                      + car_cap_size
                      + total_dis_size)
        
        self.policy = nn.Sequential(
            nn.Linear(input_size_action, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, input, critic=True):
        if critic:
            return self.value_functon(input)
        return self.policy(input)


class A2C:
    def __init__(self, gnn_model, transformer_model, num_epoches,
                 lr, hiden_dim, output_dim,
                 discount, graph_embedding_size, node_embedding_size):

        self.gnn_model = gnn_model
        self.transformer_model = transformer_model
        self.num_epoches = num_epoches
        self.discount = discount

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.used = set()

        self.A2C_network = A2CNetwork(graph_embedding_size, node_embedding_size,
                                       1, 1, hiden_dim, output_dim)
        

        self.gnn_model.to(self.device)
        self.transformer_model.to(self.device)
        self.target_model.to(self.device)
        self.explore_model.to(self.device)
        

        self.optimizer = torch.optim.Adam(
            itertools.chain(
                self.A2C_network.parameters(),
                self.gnn_model.parameters(),
                self.transformer_model.parameters()
            ),
            lr=lr
        )
        self.criterion = torch.nn.MSELoss()

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

        with torch.no_grad():

            cap_tensor = torch.tensor([route["capacity"]], dtype=torch.float32, device=self.device)
            dis_tensor = torch.tensor([route["total_distance"]], dtype=torch.float32, device=self.device)

            prob_lsit = []
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


                i_th_prob = self.A2C_network(input_policy, critic=False)

                prob_lsit.append(torch.tensor(i_th_prob))
                actions.append(i)


            tensor_list = torch.stack(prob_lsit)
            total_sum = torch.sum(tensor_list)
            scaled_tensor = tensor_list / total_sum

        return scaled_tensor, actions

    def policy(self, route):

        action_probs, actions = self.policy_probabilities(route)
        chosen_action = torch.multinomial(action_probs, num_samples=1)
        chosen_action = chosen_action.item()
        return actions[chosen_action], action_probs[chosen_action], 

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

        self.gnn_input, self.mdp = create_train_env()
        self.gnn_input = self.gnn_input.to(self.device)

        for _ in range(self.num_epoches):

            self.routes = self.refresh_env()
            finished = {k: False for k in self.routes}
            self.used = set()

            while len(self.used) != self.mdp.num_nodes - 1 or not all(finished.values()):
                self.graph_embedding, self.node_embedding = self.compute_embeddings(requires_grad=True)
                for route in self.routes:

                    current_value_fun = self.value_function(route)
                    action, action_prob = self.policy(route=route)

                    reward = -1 * self.mdp.distance_matrix[route["current_node"]][action]
                    self.used.add(action)

                    route["path"].append(action)
                    route["capacity"] -= self.mdp.node_capacity[action]
                    route["total_distance"] -= reward
                    route["current_node"] = action

                    next_value_fun = self.value_function(route)

                    
                    y = reward + next_value_fun
                    loss = self.criterion(y, current_value_fun)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()





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