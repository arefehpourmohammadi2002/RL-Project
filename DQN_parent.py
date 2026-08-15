from abc import ABC, abstractmethod

import torch.nn as nn
import torch
from copy import deepcopy

import replay_buffer as RB

INPUT_DIM = 14
class DQNNetwork(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        '''
        the input is:
        graph statics:
            - graph number of nodes -> 1
            - graph average distance -> 1
            - graph std distances -> 1
            - car capcities -> 1
            - nodes demands average -> 1
            - nodes std demands -> 1
        
        current route statics:
            - total distance -> 1
            - remaining capacity -> 1
        
        other routs statics:
            - number of remainign nodes -> 1
            - remaining demand -> 1
            - emebbed of the remaining nodes -> 1
            - other routes average total distance -> 1
        
        next node:
            - nodes capacity -> 1
            - current link to it distnace from current route -> 1
        '''

        self.input_dim = INPUT_DIM
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dqn = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(), # --> is Rely the best option
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, input):
        return self.dqn(input)

class DQN(ABC):
    def __init__(self, hidden_dim, output_dim, 
                 lr, 
                 target_update_counter, explore_update_counter):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.explore_model = DQNNetwork(hidden_dim, output_dim)
        self.target_model = DQNNetwork(hidden_dim, output_dim)
        self.target_model.load_state_dict(self.explore_model.state_dict())

        self.optimizer = torch.optim.Adam( # abetter optimizer what about the parameters 
            self.explore_model.parameters(), 
            lr =lr
        )

        self.criterion = nn.MSELoss() # better criterion

        self.target_update_limit = target_update_counter
        self.explore_update_limit = explore_update_counter
        
        self.target_update_counter = 0
        self.explore_update_counter = 0

    def get_candidate(self, mdp, route, used):
        candidates = [i for i in range(mdp.num_nodes) if i not in used]
        remaining_cap = mdp.car_capacity - mdp.capacity_marix[route["current_node"]] 

        mask = mdp.node_capacities <= remaining_cap
        candidates = [candidates[i] for i in range(len(mask)) if mask[i]]
        
        return candidates

    def add_node_to_route(self, route, next_node):
        route["path"].append(next_node)
        route["total_distance"] +=  self.mdp.diatance_matrix[route["current_node"]][next_node]
        route["capacity"] += self.mdp.capacity_matrix[route["current_node"]][next_node]
        route["current_node"] = next_node
        return route

    def edit_route(self, route, next_node): # for not distroying the original route in replay buffer
        new_route = deepcopy(route)
        return self.add_node_to_route(new_route, next_node)

    @abstractmethod
    def get_graph_statics(self, mdp, train):
        pass

    def get_unused_nodes_statics(self, mdp, unused):
        frac_unused = torch.tensor(len(unused) / mdp.num_nodes, device=self.device, dtype=float)
        ave_unused_cap = torch.tensor(sum(mdp.capacity_matrix[unused]) / len(unused), 
                                      device=self.device, dtype=float)

        return torch.cat([frac_unused, ave_unused_cap])

    def get_other_routes_static(self, routes):
        '''
        other routs statics:
            - other average used capacity -> 1
            - other routes average total distance -> 1
        '''
        sum_route_dis = 0
        sum_capacity = 0

        for route in routes:
            sum_route_dis += route["total_distance"]
            sum_capacity += route["capacity"]

        ave_other_routes_total_dis = sum_route_dis/len(routes)
        ave_other_routes_used_cap = sum_capacity/len(routes)

        ave_dis = torch.tensor(ave_other_routes_total_dis, device=self.device, dtype=float)
        ave_cap = torch.tensor(ave_other_routes_used_cap, device=self.device, dtype=float)

        return torch.cat([ave_dis, ave_cap])

    def get_current_route_statics(self, route):
        total_distance = torch.tensor(route["total_distance"], device=self.device)
        used_cap = torch.tensor(route["capacity"], device=self.device)
        return torch.cat([total_distance, used_cap])

    def get_next_node_statics(self, mdp, route, next_node):
        '''                
        next node:
            - nodes capacity -> 1
            - current link to it distnace from current route -> 1
        '''
        sum_dis = torch.tensor(sum(mdp.distance_matrix[i][next_node] 
                                   for i in range(mdp.num_nodes) if i != next_node),
                                   device=self.device, dtype=float)
        dis = torch.tensor(mdp.distance_matrix[route["current_node"]][next_node])
        cap = torch.tensor(mdp.capacit_matrix[next_node], device=self.device, dtype=float)

        return torch.cat([sum_dis, cap, dis])

    def create_input(self, mdp, routes, next_node, route, unused, train):
        graph_statics = self.get_graph_statics(mdp, train)
        unused_nodes_statics = self.get_unused_nodes_statics(mdp, unused)
        other_routes_statics = self.get_other_routes_static(mdp, routes)
        current_route_statics = self.get_current_route_statics(mdp, route)
        next_nodes_statics = self.get_next_node_statics()

        graph_statics = torch.tensor(graph_statics, device=self.device)
        unused_nodes_statics = torch.tensor(unused_nodes_statics, device=self.device)
        other_routes_statics = torch.tensor(other_routes_statics, device=self.device)
        current_route_statics = torch.tensor(current_route_statics, device=self.device)
        next_nodes_statics = torch.tensor(next_nodes_statics, device=self.device)

        return torch.cat([graph_statics, unused_nodes_statics, 
                        other_routes_statics, current_route_statics,
                        next_nodes_statics])

    def target_resutl(self, mdps, new_routes, average_dis, std_dis,
                    car_caps, node_ave_demands, node_std_demands): ########### needs work

        input_vec = []
        for route in new_routes:
            input_per_route = self.create_input(mdps, routes, average_dis, std_dis,
                        car_caps, node_ave_demands, node_std_demands)
            input_vec.append(input_per_route)

        target_model_input = torch.tensor(input_vec, device=self.device)
        return self.target_model(target_model_input)

    def train_step(self):
        (mdps, routes, route, next_node, rewards,
        average_dis, std_dis, car_caps, 
        node_ave_demands, node_std_demands, q_values) = self.replay_buffer.sample(self.batch_size)

        new_routes = []
        for k in range(len(routes)):
            new_route = self.edit_route(route[k], next_node=next_node[k])
            new_routes.append(new_route)

        target_answer = self.target_resutl(mdps, new_routes, average_dis, std_dis, 
                                           car_caps, node_ave_demands, node_std_demands)
        y = rewards + self.discount * target_answer

        loss = self.criterion(q_values, y)
        self.optimizer.zero_grad
        loss.backward()
        self.optimizer.step()

    def training_episode(self, route, next_node, q_value):
        reward = -1 * self.mdp.diatance_matrix[route["current_node"]][next_node]

        self.replay_buffer.insert(self.mdp, self.routes, route, next_node, reward,
                                  self.average_dis, self.std_dis, self.car_cap, 
                                  self.node_ave_demands, self.node_std_demands, q_value)
        
        self.explore_update_counter += 1
        if self.explore_update_counter >= self.explore_update_limit:
            self.train_step()

        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_limit:
            self.target_model.load_state_dict(self.explore_model.state_dict())

    def policy(self, train): ## needs work
        input = []
        if train and random.random() <= self.epsilon:
            return 
        for route in self.routes:
            candidates = self.get_candidate(route)
            for candidate in candidates:
                input.append(self.create_input(self.mdp, self.routes, route, candidate))

        explore_input = torch.tensor(input, device=self.device)
        q_values = self.explore_model(explore_input)
        index = q_values.argmax()
        
        return q_values[index], self.routes[index // len(candidates)], candidates[index % index]
              
    def run_episode(self, train=True):
        self.routes, self.mdp, self.used = self.new_episode()
        finished = [False] * len(self.mdp.num_cars)

        while not all(finished):
            q_value, route, next_node = self.policy(train)
            if train:
                self.training_step(deepcopy(route), next_node, q_value)
            self.add_route(route, next_node)

    def prepare_replay_buffer(self):
        self.replay_buffer = RB(self.RB_capacity)
        self.replay_buffer.full_buffer(self.num_first_samples)
        
    def DQN_train(self):
        self.prepare_replay_buffer()

        for epoch in range(len(self.num_epoches)):
            self.run_episode()
            print(epoch)
            
    