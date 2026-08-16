import matplotlib.pyplot as plt
import torch
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


gnn_trans_dqn = GNNTransDQN(
        input_dim_gnn=3,
        gnn_hidden_dim=16,
        gnn_output_dim=11,
        large_value=100,
        num_layers=3,
        num_heads=2,
        model_dim=8,
        FF_hidden_dim=24,
        input_dim=29,

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



gnn_trans_dqn.DQN_train()
torch.save(
    {
        "explore_model": gnn_trans_dqn.explore_model.state_dict(),
        "gnn": gnn_trans_dqn.gnn.state_dict(),
        "transformer": gnn_trans_dqn.transforemer.state_dict(),
    },
    config["comparison"]["checkpoints"]["gnn_transformer_dqn"],
)



heuristic_trials = []
gnn_trans_dqn_trials = []

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

        gnn_trans_dqn_total_dis, gnn_trans_dqn_routes = gnn_trans_dqn.evaluate(test_mdp)
        if len(gnn_trans_dqn.used) == test_mdp.num_nodes:
            gnn_trans_dqn_trials.append(gnn_trans_dqn_total_dis)
            print(gnn_trans_dqn_routes)
        else:
            raise RuntimeError("GNNTransDQN failed to visit every customer")

plt.plot(heuristic_trials, label="heuristic")
plt.plot(gnn_trans_dqn_trials, label="gnn trans dqn + local search")
plt.legend()
plt.savefig("gnn_trans_node.png")
