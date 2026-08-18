from abc import ABC, abstractmethod
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from DQN_parent import DQN
from MDP import MDP


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.actor(x)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.critic(x)


class A2C(ABC):
    """Lightweight A2C base compatible with the DQN-style agents."""
    def __init__(self,
                 actor_input_dim,
                 critic_input_dim,
                 hidden_dim,
                 lr,
                 discount,
                 entropy_coef,
                 value_loss_coef,
                 episodes_per_update,
                 num_epoches,
                 max_num_nodes,
                 min_num_nodes,
                 max_num_cars,
                 min_num_cars,
                 cars_capacity,
                 depot_num,
                 min_distance,
                 max_distance,
                 min_node_dem,
                 max_node_dem,
                 max_grad_norm,
                 ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(input_dim=actor_input_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic = CriticNetwork(input_dim=critic_input_dim, hidden_dim=hidden_dim).to(self.device)

        self.lr = lr
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        self.discount = discount
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.episodes_per_update = episodes_per_update
        self.num_epoches = num_epoches
        self.max_grad_norm = max_grad_norm

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

        self.actor_loss_history = []
        self.critic_loss_history = []

    @staticmethod
    def distance_scale(mdp):
        return max(float(sum(mdp.distance_matrix_ave)) / max(mdp.num_nodes, 1), 1e-6)

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
        while True:
            self.num_nodes = random.randint(self.min_num_nodes, self.max_num_nodes)
            self.num_cars = random.randint(self.min_num_cars, self.max_num_cars)
            mdp = MDP(num_nodes=self.num_nodes, depot_num=self.depot_num, num_cars=self.num_cars, cars_capacity=self.cars_capacity)
            mdp.build(min_distance=self.min_distance, max_distance=self.max_distance, min_node_dem=self.min_node_dem, max_node_dem=self.max_node_dem)
            demands = [mdp.node_demand[node] for node in range(mdp.num_nodes) if node != mdp.depot_num]
            capacities = [mdp.cars_capacity] * mdp.num_cars
            if DQN.can_pack_demands(demands, capacities):
                routes = [self.new_route() for _ in range(self.num_cars)]
                return routes, mdp, {self.depot_num}

    def new_route(self):
        return {"path": [self.depot_num], "total_distance": 0.0, "capacity": 0.0, "current_node": self.depot_num}

    def get_candidate(self, mdp, route, used):
        remaining_cap = mdp.cars_capacity - route["capacity"]
        return [node for node in range(mdp.num_nodes) if node not in used and node != mdp.depot_num and mdp.node_demand[node] <= remaining_cap + 1e-6]

    def action_keeps_solution_feasible(self, mdp, routes, route_idx, next_node, used):
        remaining_demands = [mdp.node_demand[node] for node in range(mdp.num_nodes) if node not in used and node != next_node and node != mdp.depot_num]
        remaining_capacities = [mdp.cars_capacity - route["capacity"] - (mdp.node_demand[next_node] if idx == route_idx else 0.0) for idx, route in enumerate(routes)]
        return DQN.can_pack_demands(remaining_demands, remaining_capacities)

    def collect_candidates(self):
        candidate_map = {}
        for route_idx, route in enumerate(self.routes):
            candidates = [node for node in self.get_candidate(self.mdp, route, self.used) if self.action_keeps_solution_feasible(self.mdp, self.routes, route_idx, node, self.used)]
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
        distance_scale = max((self.mdp.num_nodes - 1) * self.distance_scale(self.mdp), 1e-6)
        sum_route_dis = sum(r["total_distance"] for r in other_routes) / distance_scale
        sum_capacity = sum(r["capacity"] for r in other_routes) / max(self.mdp.cars_capacity, 1e-6)
        ave_dis = torch.tensor([sum_route_dis / len(other_routes)], device=self.device, dtype=torch.float32)
        ave_cap = torch.tensor([sum_capacity / len(other_routes)], device=self.device, dtype=torch.float32)
        return torch.cat([ave_dis, ave_cap])

    def get_current_route_statics(self, route):
        distance_scale = max((self.mdp.num_nodes - 1) * self.distance_scale(self.mdp), 1e-6)
        total_distance = torch.tensor([route["total_distance"] / distance_scale], device=self.device, dtype=torch.float32)
        used_cap = torch.tensor([route["capacity"] / max(self.mdp.cars_capacity, 1e-6)], device=self.device, dtype=torch.float32)
        return torch.cat([total_distance, used_cap])

    def get_state_statics(self, routes):
        if not routes:
            return torch.zeros(3, device=self.device, dtype=torch.float32)
        route_stats = torch.stack([self.get_current_route_statics(route) for route in routes])
        mean_stats = route_stats.mean(dim=0)
        open_routes = sum(route["current_node"] != self.depot_num for route in routes)
        open_fraction = torch.tensor([open_routes / len(routes)], device=self.device, dtype=torch.float32)
        return torch.cat([mean_stats, open_fraction])

    def close_finished_routes(self, training):
        del training
        for route in self.routes:
            if route["current_node"] != self.mdp.depot_num:
                if not self.get_candidate(mdp=self.mdp, route=route, used=self.used):
                    self.add_node_to_route(mdp=self.mdp, route=route, next_node=self.mdp.depot_num, used=self.used)

    def add_route(self, route_idx, next_node, training):
        self.routes[route_idx] = self.add_node_to_route(mdp=self.mdp, route=self.routes[route_idx], next_node=next_node, used=self.used)
        self.close_finished_routes(training=training)

    def policy(self, train, candidate_map):
        graph_statics = self.get_graph_statics(self.mdp, train)
        unused_nodes_statics = self.get_unused_nodes_statics(self.mdp, self.used)

        actor_input = []
        index_map = []
        for route_idx, candidates in candidate_map.items():
            route = self.routes[route_idx]
            current_route_statics = self.get_current_route_statics(route)
            other_routes_statics = self.get_other_routes_static(self.routes, route)
            for candidate in candidates:
                next_node_statics = self.get_next_node_statics(self.mdp, route, candidate)
                actor_input.append(torch.cat([graph_statics, unused_nodes_statics, other_routes_statics, current_route_statics, next_node_statics]))
                index_map.append((route_idx, candidate))

        actor_input = torch.stack(actor_input)
        state_statics = self.get_state_statics(self.routes)
        critic_input = torch.cat([graph_statics, unused_nodes_statics, state_statics]).unsqueeze(0)

        if train:
            logits = self.actor(actor_input).squeeze(-1)
            value = self.critic(critic_input).squeeze()
            dist = torch.distributions.Categorical(logits=logits)
            index = dist.sample()
            log_prob = dist.log_prob(index)
            entropy = dist.entropy()
        else:
            with torch.no_grad():
                logits = self.actor(actor_input).squeeze(-1)
                value = self.critic(critic_input).squeeze()
            index = logits.argmax()
            log_prob = None
            entropy = None

        route_idx, next_node = index_map[index.item()]
        return route_idx, next_node, log_prob, entropy, value

    def run_episode(self, train=True):
        if train:
            self.routes, self.mdp, self.used = self.new_episode()
        candidate_map = self.collect_candidates()

        steps = []
        while candidate_map:
            route_idx, next_node, log_prob, entropy, value = self.policy(train=train, candidate_map=candidate_map)
            route = self.routes[route_idx]
            reward = -self.marginal_cost(self.mdp, route, next_node) / self.distance_scale(self.mdp)

            if train:
                steps.append([log_prob, entropy, value, reward])

            self.add_route(route_idx, next_node, training=train)
            candidate_map = self.collect_candidates()

        if not train or not steps:
            return []

        running_return = 0.0
        for step in reversed(steps):
            running_return = step[3] + self.discount * running_return
            step[3] = running_return

        return steps

    def update(self, steps):
        if not steps:
            return

        log_probs = torch.stack([step[0] for step in steps])
        entropies = torch.stack([step[1] for step in steps])
        values = torch.stack([step[2] for step in steps])
        returns = torch.tensor([step[3] for step in steps], device=self.device, dtype=torch.float32)

        advantage = returns - values.detach()

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = (returns - values).pow(2).mean()
        entropy_bonus = entropies.mean()

        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), max_norm=self.max_grad_norm)
        self.optimizer.step()

        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())

    def A2C_train(self):
        buffer = []
        for epoch in range(self.num_epoches):
            buffer.extend(self.run_episode(train=True))

            if (epoch + 1) % self.episodes_per_update == 0:
                self.update(buffer)
                buffer = []

            if (epoch + 1) % max(self.num_epoches // 20, 1) == 0:
                print(epoch)

        if buffer:
            self.update(buffer)

        plt.plot(self.actor_loss_history, label="actor loss")
        plt.plot(self.critic_loss_history, label="critic loss")
        plt.legend()
        plt.savefig("a2c_loss.png")
        plt.close()

    def evaluate(self, mdp):
        self.mdp = mdp
        self.routes = [self.new_route() for _ in range(mdp.num_cars)]
        self.used = {self.depot_num}
        self.run_episode(train=False)
        total_distance = sum(route["total_distance"] for route in self.routes)
        return total_distance, self.routes
