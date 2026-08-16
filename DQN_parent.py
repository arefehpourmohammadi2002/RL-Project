from abc import ABC, abstractmethod
import random
import torch.nn as nn
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import replay_buffer as RB
from MDP import MDP


class DQNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dqn = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, input):
        return self.dqn(input)


class DQN(ABC):
    def __init__(self, input_dim, hidden_dim, output_dim,
                lr, lr_decay, min_lr, target_update_counter, explore_update_counter,
                discount, epsilon, epsilon_decay, min_epsilon,
                batch_size, RB_capacity, num_first_samples, num_epoches,
                max_num_nodes, min_num_nodes, max_num_cars, min_num_cars,
                cars_capacity, depot_num,
                min_distance,
                max_distance,
                min_node_dem,
                max_node_dem, max_grad_norm):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.explore_model = DQNNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(self.device)
        self.target_model = DQNNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(self.device)
        self.target_model.load_state_dict(self.explore_model.state_dict())

        self.lr = lr
        self.lr_decay = lr_decay
        self.min_lr = min_lr

        self.criterion = nn.MSELoss()

        self.target_update_limit = target_update_counter
        self.explore_update_limit = explore_update_counter

        self.target_update_counter = 0
        self.explore_update_counter = 0

        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.RB_capacity = RB_capacity
        self.num_first_samples = num_first_samples
        self.num_epoches = num_epoches

        self.depot_num = depot_num
        self.max_num_nodes = max_num_nodes
        self.min_num_nodes = min_num_nodes
        self.max_num_cars = max_num_cars
        self.min_num_cars = min_num_cars
        self.cars_capacity = cars_capacity
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.min_node_dem = min_node_dem
        self.max_node_dem = max_node_dem

        self.routes = None
        self.mdp = None
        self.used = None
        self.replay_buffer = None

        self.max_grad_norm = max_grad_norm

        self.loss_history = []
        self.lr_history = []
        self.epsilon_history = []

    def new_episode(self):
        self.num_nodes = random.randint(self.min_num_nodes, self.max_num_nodes)
        self.num_cars = random.randint(self.min_num_cars, self.max_num_cars)
        cars_capacity = random.uniform(1, self.cars_capacity)

        mdp = MDP(num_nodes=self.num_nodes, depot_num=self.depot_num,
                 num_cars=self.num_cars, cars_capacity=cars_capacity)

        mdp.build(min_distance=self.min_distance, max_distance=self.max_distance,
                 min_node_dem=self.min_node_dem, max_node_dem=self.max_node_dem)

        routes = [self.new_route() for _ in range(self.num_cars)]
        used = {self.depot_num}

        return routes, mdp, used

    def new_route(self):
        return {
            "path": [self.depot_num],
            "total_distance": 0.0,
            "capacity": 0.0,
            "current_node": self.depot_num
        }

    def get_candidate(self, mdp, route, used):
        remaining_cap = mdp.cars_capacity - route["capacity"]
        return [i for i in range(mdp.num_nodes) if i not in used and mdp.node_demand[i] <= remaining_cap]

    def collect_candidates(self):
        candidate_map = {}
        for route_idx, route in enumerate(self.routes):
            candidates = self.get_candidate(mdp=self.mdp, route=route, used=self.used)
            if candidates:
                candidate_map[route_idx] = candidates
        return candidate_map

    def add_node_to_route(self, mdp, route, next_node, used):
        route["path"].append(next_node)
        route["total_distance"] += mdp.distance_matrix[route["current_node"]][next_node]
        route["capacity"] += mdp.node_demand[next_node]
        route["current_node"] = next_node

        if next_node != mdp.depot_num:
            used.add(next_node)

        return route

    def edit_route(self, mdp, route, next_node, used):
        new_route = deepcopy(route)
        return self.add_node_to_route(mdp=mdp, route=new_route, next_node=next_node, used=used)

    def get_unused_nodes(self, mdp, used):
        return [i for i in range(mdp.num_nodes) if i not in used]

    @abstractmethod
    def get_graph_statics(self, mdp, train):
        pass

    @abstractmethod
    def get_unused_nodes_statics(self, mdp, used):
        pass

    @abstractmethod
    def get_next_node_statics(self, mdp, route, next_node):
        pass

    def get_other_routes_static(self, routes, route):
        other_routes = [r for r in routes if r is not route]

        if not other_routes:
            return torch.zeros(2, device=self.device, dtype=torch.float32)

        sum_route_dis = sum(r["total_distance"] for r in other_routes)
        sum_capacity = sum(r["capacity"] for r in other_routes)

        ave_dis = torch.tensor([sum_route_dis / len(other_routes)], device=self.device, dtype=torch.float32)
        ave_cap = torch.tensor([sum_capacity / len(other_routes)], device=self.device, dtype=torch.float32)

        return torch.cat([ave_dis, ave_cap])

    def get_current_route_statics(self, route):
        total_distance = torch.tensor([route["total_distance"]], device=self.device, dtype=torch.float32)
        used_cap = torch.tensor([route["capacity"]], device=self.device, dtype=torch.float32)
        return torch.cat([total_distance, used_cap])

    def create_input(self, mdp, routes, route, next_node, used, train):
        graph_statics = self.get_graph_statics(mdp, train)
        unused_nodes_statics = self.get_unused_nodes_statics(mdp, used)
        next_nodes_statics = self.get_next_node_statics(mdp, route, next_node)
        other_routes_statics = self.get_other_routes_static(routes, route)
        current_route_statics = self.get_current_route_statics(route)

        return torch.cat([graph_statics, unused_nodes_statics,
                        other_routes_statics, current_route_statics,
                        next_nodes_statics])

    def explore_result(self, rb_objs, train):
        explore_input = []
        for rb_obj in rb_objs:
            route = rb_obj.routes[rb_obj.route_idx]
            input_per_obj = self.create_input(mdp=rb_obj.mdp, routes=rb_obj.routes, route=route,
                                              next_node=rb_obj.next_node, used=rb_obj.used, train=train)
            explore_input.append(input_per_obj)

        explore_input = torch.stack(explore_input)
        return self.explore_model(explore_input)

    def create_target_input(self, rb_obj):
        mdp = rb_obj.mdp
        routes = deepcopy(rb_obj.routes)
        used = deepcopy(rb_obj.used)

        updated_route = self.add_node_to_route(mdp=mdp, route=deepcopy(routes[rb_obj.route_idx]),
                                               next_node=rb_obj.next_node, used=used)
        routes[rb_obj.route_idx] = updated_route

        graph_statics = self.get_graph_statics(mdp, False)
        unused_nodes_statics = self.get_unused_nodes_statics(mdp, used)

        target_model_input = []
        for route in routes:
            candidates = self.get_candidate(mdp=mdp, route=route, used=used)
            if not candidates:
                continue

            current_route_statics = self.get_current_route_statics(route)
            other_routes_statics = self.get_other_routes_static(routes, route)

            for candid in candidates:
                next_nodes_statics = self.get_next_node_statics(mdp, route, candid)
                target_model_input.append(torch.cat([graph_statics, unused_nodes_statics,
                                other_routes_statics, current_route_statics,
                                next_nodes_statics]))

        if not target_model_input:
            return None

        return torch.stack(target_model_input)

    def target_result(self, rb_objs):
        target_inputs = [self.create_target_input(rb_obj) for rb_obj in rb_objs]
        split_sizes = [0 if target_input is None else target_input.size(0) for target_input in target_inputs]

        non_empty_inputs = [target_input for target_input in target_inputs if target_input is not None]

        if non_empty_inputs:
            combined_input = torch.cat(non_empty_inputs, dim=0)
            with torch.no_grad():
                combined_explore_q = self.explore_model(combined_input).squeeze(-1)
                combined_target_q = self.target_model(combined_input).squeeze(-1)
        else:
            combined_explore_q = torch.zeros(0, device=self.device)
            combined_target_q = torch.zeros(0, device=self.device)

        answer_vec = []
        offset = 0
        for size in split_sizes:
            if size == 0:
                answer_vec.append(torch.tensor(0.0, device=self.device))
            else:
                best_index = combined_explore_q[offset:offset + size].argmax()
                answer_vec.append(combined_target_q[offset:offset + size][best_index])
                offset += size

        return torch.stack(answer_vec)

    def train_step(self):
        rb_objs = self.replay_buffer.sample(self.batch_size)

        target_answer = self.target_result(rb_objs)
        rewards = torch.tensor([rb_obj.reward for rb_obj in rb_objs], device=self.device, dtype=torch.float32)
        y = rewards + self.discount * target_answer

        explore_result = self.explore_result(rb_objs, train=True).squeeze(-1)

        loss = self.criterion(explore_result, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.explore_model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        self.loss_history.append(loss.item())

    def training_episode(self, route_idx, next_node):
        route = self.routes[route_idx]
        reward = -1 * self.mdp.distance_matrix[route["current_node"]][next_node]

        self.replay_buffer.insert(mdp=self.mdp, routes=self.routes, route_idx=route_idx,
                                  next_node=next_node, reward=reward, used=self.used)

        self.explore_update_counter += 1
        if self.explore_update_counter >= self.explore_update_limit and self.replay_buffer.size() >= self.batch_size:
            self.train_step()
            self.explore_update_counter = 0

        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_limit:
            self.target_model.load_state_dict(self.explore_model.state_dict())
            self.target_update_counter = 0

    def policy(self, train, candidate_map):
        if train and random.random() <= self.epsilon:
            all_pairs = [(route_idx, candidate)
                        for route_idx, candidates in candidate_map.items()
                        for candidate in candidates]
            return random.choice(all_pairs)

        graph_statics = self.get_graph_statics(self.mdp, False)
        unused_nodes_statics = self.get_unused_nodes_statics(self.mdp, self.used)

        input = []
        index_map = []
        for route_idx, candidates in candidate_map.items():
            route = self.routes[route_idx]
            current_route_statics = self.get_current_route_statics(route)
            other_routes_statics = self.get_other_routes_static(self.routes, route)

            for candidate in candidates:
                next_nodes_statics = self.get_next_node_statics(self.mdp, route, candidate)
                input.append(torch.cat([graph_statics, unused_nodes_statics,
                                other_routes_statics, current_route_statics,
                                next_nodes_statics]))
                index_map.append((route_idx, candidate))

        explore_input = torch.stack(input)
        with torch.no_grad():
            q_values = self.explore_model(explore_input).squeeze(-1)
        index = q_values.argmax().item()

        return index_map[index]

    def close_finished_routes(self, training):
        for route_idx, route in enumerate(self.routes):
            if route["current_node"] != self.mdp.depot_num:
                if not self.get_candidate(mdp=self.mdp, route=route, used=self.used):
                    reward = -1 * self.mdp.distance_matrix[route["current_node"]][self.mdp.depot_num]

                    if training:
                        self.replay_buffer.insert(mdp=self.mdp, routes=self.routes, route_idx=route_idx,
                                                  next_node=self.mdp.depot_num, reward=reward, used=self.used)

                    self.add_node_to_route(mdp=self.mdp, route=route, next_node=self.mdp.depot_num, used=self.used)

    def add_route(self, route_idx, next_node, training):
        self.routes[route_idx] = self.add_node_to_route(mdp=self.mdp, route=self.routes[route_idx],
                                                         next_node=next_node, used=self.used)
        self.close_finished_routes(training=training)

    def run_episode(self, train=True):
        if train:
            self.routes, self.mdp, self.used = self.new_episode()
        candidate_map = self.collect_candidates()

        while candidate_map:
            route_idx, next_node = self.policy(train=train, candidate_map=candidate_map)
            if train:
                self.training_episode(route_idx, next_node)
            self.add_route(route_idx, next_node, training=train)
            candidate_map = self.collect_candidates()

    def run_random_episode(self, max_sample_of_one_env):
        self.routes, self.mdp, self.used = self.new_episode()
        candidate_map = self.collect_candidates()

        i = 0
        while candidate_map and i <= max_sample_of_one_env:
            route_idx = random.choice(list(candidate_map.keys()))
            next_node = random.choice(candidate_map[route_idx])

            route = self.routes[route_idx]
            reward = -1 * self.mdp.distance_matrix[route["current_node"]][next_node]
            self.replay_buffer.insert(mdp=self.mdp, routes=self.routes, route_idx=route_idx,
                                      next_node=next_node, reward=reward, used=self.used)

            self.add_route(route_idx, next_node, training=True)
            candidate_map = self.collect_candidates()

            i += 1

    def full_replay_buffer(self, num_first_samples):
        max_sample_of_one_env = random.randint(2, max(num_first_samples, num_first_samples//10))
        while self.replay_buffer.size() < num_first_samples:
            self.run_random_episode(max_sample_of_one_env)
        for snap in self.replay_buffer.replay_buffer:
            print(snap.route_idx, snap.next_node, snap.reward)

    def prepare_replay_buffer(self): 
        self.replay_buffer = RB.ReplayBuffer(capacity=self.RB_capacity)
        self.full_replay_buffer(num_first_samples=self.num_first_samples)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    def decay_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = max(self.min_lr, param_group["lr"] * self.lr_decay)
            self.lr = param_group["lr"]
        self.lr_history.append(self.lr)

    def DQN_train(self):
        self.prepare_replay_buffer()

        for epoch in range(self.num_epoches):
            self.run_episode(train=True)
            self.decay_epsilon()
            self.decay_lr()
            print(epoch)

        plt.plot(self.loss_history)
        plt.savefig('loss.png')
        plt.close()
        plt.plot(self.lr_history, label="lr")
        plt.plot(self.epsilon_history, label="epsilon")
        plt.savefig('lr_epsilon.png')
        plt.close()

    def evaluate(self, mdp):
        self.mdp = mdp
        self.routes = [self.new_route() for _ in range(mdp.num_cars)]
        self.used = {self.depot_num}
        self.run_episode(train=False)
        total_distance = sum(route["total_distance"] for route in self.routes)

        return total_distance, self.routes