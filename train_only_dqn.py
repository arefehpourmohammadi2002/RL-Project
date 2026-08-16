import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from heuristic import ClarkeWrightSavings
from MDP import MDP
from onlyDQN.only_DQN import OnlyDQN


def total_dis(routes, mdp):
    overall_total_dis = 0
    for route in routes:
        route = route[0]
        route_distance = mdp.distance_matrix[mdp.depot_num][route[0]]
        for i in range(len(route) - 1):
            route_distance += mdp.distance_matrix[route[i]][route[i + 1]]
        route_distance += mdp.distance_matrix[route[-1]][mdp.depot_num]
        overall_total_dis += route_distance
    return overall_total_dis

with open("conf.yaml", "r") as file:
    config = yaml.safe_load(file)

output_dim = config["output_dim"]
hidden_dim = config["hidden_dim"]
lr = config["lr"]
lr_decay = config["lr_decay"]
min_lr = config["min_lr"]
target_update_counter = config["target_update_counter"]
explore_update_counter = config["explore_update_counter"]
discount = config["discount"]
epsilon = config["epsilon"]
epsilon_decay = config["epsilon_decay"]
min_epsilon = config["min_epsilon"]
batch_size = config["batch_size"]
RB_capacity = config["RB_capacity"]
num_first_samples = config["num_first_samples"]
num_epoches = config["num_epoches"]
max_num_nodes = config["max_num_nodes"]
min_num_nodes = config["min_num_nodes"]
depot_num = config["depot_num"]
max_num_cars = config["max_num_cars"]
min_num_cars = config["min_num_cars"]
cars_capacity = config["cars_capacity"]
min_distance = config["min_distance"]
max_distance = config["max_distance"]
min_node_dem = config["min_node_dem"]
max_node_dem = config["max_node_dem"]
max_grad_norm = config["max_grad_norm"]

with open("conf.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

seed = config["comparison"]["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


only_dqn = OnlyDQN(
    input_dim=OnlyDQN.INPUT_DIM,
    output_dim=config["output_dim"],
    hidden_dim=config["hidden_dim"],
    lr=config["lr"],
    lr_decay=config["lr_decay"],
    min_lr=config["min_lr"],
    target_update_counter=config["target_update_counter"],
    explore_update_counter=config["explore_update_counter"],
    discount=config["discount"],
    epsilon=config["epsilon"],
    epsilon_decay=config["epsilon_decay"],
    min_epsilon=config["min_epsilon"],
    batch_size=config["batch_size"],
    RB_capacity=config["RB_capacity"],
    num_first_samples=config["num_first_samples"],
    num_epoches=config["num_epoches"],
    max_num_nodes=config["max_num_nodes"],
    min_num_nodes=config["min_num_nodes"],
    depot_num=config["depot_num"],
    max_num_cars=config["max_num_cars"],
    min_num_cars=config["min_num_cars"],
    cars_capacity=config["cars_capacity"],
    min_distance=config["min_distance"],
    max_distance=config["max_distance"],
    min_node_dem=config["min_node_dem"],
    max_node_dem=config["max_node_dem"],
    max_grad_norm=config["max_grad_norm"],
)

only_dqn.DQN_train()
torch.save(
    {"explore_model": only_dqn.explore_model.state_dict()},
    config["comparison"]["checkpoints"]["only_dqn"],
)

heuristic_trials = []
only_dqn_trials = []

for n_nodes in range(min_num_nodes, min(max_num_nodes + 1, min_num_nodes + 5)):
    for n_cars in range(min_num_cars, max_num_cars + 1):
        test_mdp = MDP(n_nodes, depot_num, n_cars, cars_capacity)
        test_mdp.fill_distance_matrix(min_distance, max_distance)
        test_mdp.fill_node_dem_matrix(min_node_dem, max_node_dem)

        cws = ClarkeWrightSavings(test_mdp)
        feasible = cws.CWS_solve()
        if feasible:
            heuristic_trial_dis = total_dis(cws.list_routes, test_mdp)
            heuristic_trials.append(heuristic_trial_dis)
            print(cws.list_routes)
        else:
            continue


        dqn_total_dis, dqn_routes = only_dqn.evaluate(test_mdp)
        if len(only_dqn.used) == test_mdp.num_nodes:
            only_dqn_trials.append(dqn_total_dis)
            print(dqn_routes)
        else:
            raise RuntimeError("TransformerDQN failed to visit every customer")


plt.plot(heuristic_trials, label="heuristic")
plt.plot(only_dqn_trials, label="transformer dqn + local search")

plt.legend()
plt.savefig("trans_node.png")