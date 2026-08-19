from abc import ABC, abstractmethod
from copy import deepcopy
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from DQN_parent import StateActionNet
from MDP import MDP


class ActorNetwork(nn.Module):
    """Scores each feasible (vehicle, next-customer) action."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs):
        return self.net(inputs).squeeze(-1)


class CriticNetwork(nn.Module):
    """Estimates V(s) from the common state representation."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs):
        return self.net(inputs).squeeze(-1)


class A2C(ABC):
    """Shared on-policy A2C routing implementation.

    Its environment and feature interface intentionally follow ``DQN`` so the
    four architectures can be compared on exactly the same routing problem.
    """

    def __init__(
        self, input_dim, hidden_dim, output_dim, lr, lr_decay, min_lr,
        target_update_counter, explore_update_counter, discount, epsilon,
        epsilon_decay, min_epsilon, batch_size, RB_capacity,
        num_first_samples, num_epoches, max_num_nodes, min_num_nodes,
        max_num_cars, min_num_cars, cars_capacity, depot_num, min_distance,
        max_distance, min_node_dem, max_node_dem, max_grad_norm,
        action_net_input_dim, action_net_hidden_dim, action_net_output_dim,
        state_net_input_dim, state_net_hidden_dim, state_net_output_dim,
        value_coef=0.5, entropy_coef=0.01,
    ):
        # DQN-only arguments remain accepted so all train scripts share config.
        del output_dim, target_update_counter, explore_update_counter
        del epsilon, epsilon_decay, min_epsilon, batch_size, RB_capacity
        del num_first_samples

        if state_net_input_dim + action_net_input_dim != input_dim:
            raise ValueError("state and action input dimensions must sum to input_dim")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_net = StateActionNet(
            state_net_input_dim, state_net_hidden_dim, state_net_output_dim
        ).to(self.device)
        self.action_net = StateActionNet(
            action_net_input_dim, action_net_hidden_dim, action_net_output_dim
        ).to(self.device)
        self.actor = ActorNetwork(
            state_net_output_dim + action_net_output_dim, hidden_dim
        ).to(self.device)
        self.critic = CriticNetwork(state_net_output_dim, hidden_dim).to(self.device)

        self.lr = lr
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.discount = discount
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
        self.max_grad_norm = max_grad_norm
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.routes = None
        self.mdp = None
        self.used = None
        self.loss_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.lr_history = []

    @staticmethod
    def can_pack_demands(demands, capacities):
        items = sorted((float(x) for x in demands), reverse=True)
        remaining = [float(x) for x in capacities]
        if not items:
            return True
        if not remaining or items[0] > max(remaining) + 1e-6:
            return False
        if sum(items) > sum(remaining) + 1e-6:
            return False
        for item in items:
            choices = [(capacity - item, i) for i, capacity in enumerate(remaining)
                       if capacity + 1e-6 >= item]
            if not choices:
                return False
            _, index = min(choices)
            remaining[index] -= item
        return True

    def distance_scale(self, mdp):
        return max(float(sum(mdp.distance_matrix_ave)) / max(mdp.num_nodes, 1), 1e-6)

    def new_route(self):
        return {"path": [self.depot_num], "total_distance": 0.0,
                "capacity": 0.0, "current_node": self.depot_num}

    def new_episode(self):
        while True:
            num_nodes = random.randint(self.min_num_nodes, self.max_num_nodes)
            num_cars = random.randint(self.min_num_cars, self.max_num_cars)
            mdp = MDP(num_nodes, self.depot_num, num_cars, self.cars_capacity)
            mdp.build(self.min_distance, self.max_distance,
                      self.min_node_dem, self.max_node_dem)
            demands = [mdp.node_demand[i] for i in range(mdp.num_nodes)
                       if i != mdp.depot_num]
            if self.can_pack_demands(demands, [mdp.cars_capacity] * mdp.num_cars):
                return [self.new_route() for _ in range(num_cars)], mdp, {self.depot_num}

    def get_candidate(self, mdp, route, used):
        capacity = mdp.cars_capacity - route["capacity"]
        return [i for i in range(mdp.num_nodes) if i not in used
                and i != mdp.depot_num and mdp.node_demand[i] <= capacity + 1e-6]

    def action_keeps_solution_feasible(self, mdp, routes, route_idx, next_node, used):
        demands = [mdp.node_demand[i] for i in range(mdp.num_nodes)
                   if i not in used and i not in (next_node, mdp.depot_num)]
        capacities = [mdp.cars_capacity - route["capacity"]
                      - (mdp.node_demand[next_node] if i == route_idx else 0.0)
                      for i, route in enumerate(routes)]
        return self.can_pack_demands(demands, capacities)

    def collect_candidates(self):
        result = {}
        for route_idx, route in enumerate(self.routes):
            candidates = [node for node in self.get_candidate(self.mdp, route, self.used)
                          if self.action_keeps_solution_feasible(
                              self.mdp, self.routes, route_idx, node, self.used)]
            if candidates:
                result[route_idx] = candidates
        return result

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
        others = [item for item in routes if item is not route]
        if not others:
            return torch.zeros(2, device=self.device)
        distance_scale = max((self.mdp.num_nodes - 1) * self.distance_scale(self.mdp), 1e-6)
        return torch.tensor([
            sum(r["total_distance"] for r in others) / distance_scale / len(others),
            sum(r["capacity"] for r in others) / max(self.mdp.cars_capacity, 1e-6) / len(others),
        ], device=self.device, dtype=torch.float32)

    def get_current_route_statics(self, route):
        distance_scale = max((self.mdp.num_nodes - 1) * self.distance_scale(self.mdp), 1e-6)
        return torch.tensor([
            route["total_distance"] / distance_scale,
            route["capacity"] / max(self.mdp.cars_capacity, 1e-6),
        ], device=self.device, dtype=torch.float32)

    def representations(self, candidate_map, train):
        graph = self.get_graph_statics(self.mdp, train)
        unused = self.get_unused_nodes_statics(self.mdp, self.used)
        action_inputs, state_representations, index_map = [], [], []
        for route_idx, candidates in candidate_map.items():
            route = self.routes[route_idx]
            state = self.state_net(torch.cat([
                graph, unused, self.get_other_routes_static(self.routes, route)]))
            for candidate in candidates:
                action = self.action_net(torch.cat([
                    self.get_next_node_statics(self.mdp, route, candidate),
                    self.get_current_route_statics(route)]))
                action_inputs.append(torch.cat([state, action]))
                state_representations.append(state)
                index_map.append((route_idx, candidate))
        return torch.stack(action_inputs), torch.stack(state_representations), index_map

    def policy(self, train, candidate_map):
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            inputs, states, index_map = self.representations(candidate_map, train)
            distribution = torch.distributions.Categorical(logits=self.actor(inputs))
            index = distribution.sample() if train else distribution.probs.argmax()
            value = self.critic(states[index])
        action = index_map[index.item()]
        if train:
            return action, distribution.log_prob(index), value, distribution.entropy()
        return action

    def close_finished_routes(self):
        closing_cost = 0.0
        for route in self.routes:
            if route["current_node"] != self.mdp.depot_num and not self.get_candidate(
                    self.mdp, route, self.used):
                closing_cost += self.mdp.distance_matrix[route["current_node"]][self.mdp.depot_num]
                self.add_node_to_route(self.mdp, route, self.mdp.depot_num, self.used)
        return closing_cost

    def run_episode(self, train=True):
        if train:
            self.routes, self.mdp, self.used = self.new_episode()
        log_probs, values, rewards, entropies = [], [], [], []
        candidate_map = self.collect_candidates()
        while candidate_map:
            if train:
                (route_idx, next_node), log_prob, value, entropy = self.policy(True, candidate_map)
            else:
                route_idx, next_node = self.policy(False, candidate_map)
            route = self.routes[route_idx]
            cost = float(self.mdp.distance_matrix[route["current_node"]][next_node])
            self.add_node_to_route(self.mdp, route, next_node, self.used)
            cost += self.close_finished_routes()
            if train:
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(-cost / self.distance_scale(self.mdp))
                entropies.append(entropy)
            candidate_map = self.collect_candidates()
        return log_probs, values, rewards, entropies

    def train_step(self, trajectory):
        log_probs, values, rewards, entropies = trajectory
        if not rewards:
            return
        returns, running = [], torch.tensor(0.0, device=self.device)
        for reward in reversed(rewards):
            running = torch.tensor(reward, device=self.device) + self.discount * running
            returns.append(running)
        returns = torch.stack(list(reversed(returns)))
        values = torch.stack(values)
        advantages = returns - values
        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        critic_loss = advantages.square().mean()
        entropy = torch.stack(entropies).mean()
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]
             if p.grad is not None], self.max_grad_norm)
        self.optimizer.step()
        self.loss_history.append(loss.item())
        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())

    def decay_lr(self):
        for group in self.optimizer.param_groups:
            group["lr"] = max(self.min_lr, group["lr"] * self.lr_decay)
            self.lr = group["lr"]
        self.lr_history.append(self.lr)

    def A2C_train(self):
        for epoch in range(self.num_epoches):
            self.train_step(self.run_episode(train=True))
            self.decay_lr()
            print(epoch)
        plt.plot(self.loss_history, label="total")
        plt.plot(self.actor_loss_history, label="actor")
        plt.plot(self.critic_loss_history, label="critic")
        plt.legend()
        plt.savefig("A2C/results/a2c_loss.png")
        plt.close()
        plt.plot(self.lr_history)
        plt.savefig("A2C/results/a2c_lr.png")
        plt.close()

    def evaluate(self, mdp):
        self.mdp = mdp
        self.routes = [self.new_route() for _ in range(mdp.num_cars)]
        self.used = {mdp.depot_num}
        self.run_episode(train=False)
        return sum(route["total_distance"] for route in self.routes), self.routes

    def checkpoint(self):
        return {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "state_net": self.state_net.state_dict(),
            "action_net": self.action_net.state_dict(),
        }

    def load_checkpoint(self, checkpoint):
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.state_net.load_state_dict(checkpoint["state_net"])
        self.action_net.load_state_dict(checkpoint["action_net"])
