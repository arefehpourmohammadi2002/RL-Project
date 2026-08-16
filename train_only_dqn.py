import matplotlib.pyplot as plt
import yaml

from onlyDQN.only_DQN import OnlyDQN
from transformerDQN.transformer_DQN import TransformerDQN
from GNNDQN.gnn_DQN import GNNDQN
from gnntransDQN.gnntransDQN import GNNTransDQN
from heuristic import ClarkeWrightSavings
from MDP import MDP

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
target_update_counter = config["explore_update_counter"]
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

only_dqn = OnlyDQN(
        input_dim=15,

        output_dim=output_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        lr_decay=lr_decay,
        min_lr=min_lr,
        target_update_counter=target_update_counter,
        explore_update_counter=explore_update_counter,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        batch_size=batch_size,
        RB_capacity=RB_capacity,
        num_first_samples=num_first_samples,
        num_epoches=num_epoches,
        max_num_nodes=max_num_nodes,
        min_num_nodes=min_num_nodes,
        depot_num=depot_num,
        max_num_cars=max_num_cars,
        min_num_cars=min_num_cars,
        cars_capacity=cars_capacity,
        min_distance=min_distance,
        max_distance=max_distance,
        min_node_dem=min_node_dem,
        max_node_dem=max_node_dem,
        max_grad_norm=max_grad_norm
    )


only_dqn.DQN_train()

heuristic_trials = []
only_dqn_trials = []


for n_nodes in range(9, 30):
    for n_cars in range(4, 5):

        test_mdp = MDP(n_nodes, 0, n_cars, 100)
        test_mdp.fill_distance_matrix(3, 10)
        test_mdp.fill_node_dem_matrix(10, 30)

        cws = ClarkeWrightSavings(test_mdp)
        feasible = cws.CWS_solve()
        print("neeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        print(test_mdp.distance_matrix)
        print("heuristicccccccccccccccccccccccc")
        if feasible:
            heuristic_trial_dis = total_dis(cws.list_routes, test_mdp)
            heuristic_trials.append(heuristic_trial_dis)
            print(cws.list_routes)
        else:
            heuristic_trials.append(200)

        print("dqnnnnnnnnnnnnnnnn")
        dqn_total_dis, dqn_routes = only_dqn.evaluate(test_mdp)
        if len(only_dqn.used) == test_mdp.num_nodes:
            only_dqn_trials.append(dqn_total_dis)
            print(dqn_routes)
        else:
            only_dqn_trials.append(200)

plt.plot(heuristic_trials, label="heuristic")
plt.plot(only_dqn_trials, label="only dqn")
plt.legend()
plt.savefig("dqn.png")

