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
target_update_counter = config["explore_update_counter"]
explore_update_counter = config["explore_update_counter"]
discount = config["discount"]
epsilon = config["epsilon"]
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


gnn_trans_dqn = GNNTransDQN(
        input_dim_gnn=3, 
        gnn_hidden_dim=5, 
        gnn_output_dim=7, 
        large_value=100,
        num_layers=3,  
        num_heads=2, 
        model_dim=4, 
        FF_hidden_dim=10,
        input_dim=23,

        output_dim=output_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        target_update_counter=target_update_counter,
        explore_update_counter=explore_update_counter,
        discount=discount,
        epsilon=epsilon,
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
        max_node_dem=max_node_dem
    )

gnn_dqn = GNNDQN(
        input_dim_gnn=3,
        gnn_hidden_dim=5,
        gnn_output_dim=7,
        large_value=100,
        input_dim=23,

        output_dim=output_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        target_update_counter=target_update_counter,
        explore_update_counter=explore_update_counter,
        discount=discount,
        epsilon=epsilon,
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
        max_node_dem=max_node_dem
    )

only_dqn = OnlyDQN(
        input_dim=OnlyDQN.INPUT_DIM,

        output_dim=output_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        target_update_counter=target_update_counter,
        explore_update_counter=explore_update_counter,
        discount=discount,
        epsilon=epsilon,
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
        max_node_dem=max_node_dem
    )

transformer_dqn = TransformerDQN(
        num_layers=3,
        num_heads=1,
        model_dim=5,
        FF_hidden_dim=8,
        input_dim=26,

        output_dim=output_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        target_update_counter=target_update_counter,
        explore_update_counter=explore_update_counter,
        discount=discount,
        epsilon=epsilon,
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
        max_node_dem=max_node_dem
    )

gnn_trans_dqn.DQN_train()
gnn_dqn.DQN_train()
only_dqn.DQN_train()
transformer_dqn.DQN_train()

heuristic_trials = []
only_dqn_trials = []
transformer_dqn_trials = []
gnn_dqn_trials = []
gnn_trans_dqn_trials = []

for n_nodes in range(5, 20):
    for n_cars in range(4, 5):
        test_mdp = MDP(n_nodes, 0, n_cars, 100)
        test_mdp.fill_distance_matrix(3, 10)
        test_mdp.fill_node_dem_matrix(10, 30)

        cws = ClarkeWrightSavings(test_mdp)
        feasible = cws.CWS_solve()
        print("neeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeew")
        if feasible:
            heuristic_trial_dis = total_dis(cws.list_routes, test_mdp)
            heuristic_trials.append(heuristic_trial_dis)
            print(cws.list_routes)
        else:
            heuristic_trials.append(200)

        dqn_total_dis, dqn_routes = only_dqn.evaluate(test_mdp)
        if len(only_dqn.used) == test_mdp.num_nodes:
            only_dqn_trials.append(dqn_total_dis)
            print(dqn_routes)
        else:
            only_dqn_trials.append(200)

        trans_dqn_total_dis, trans_dqn_routes = transformer_dqn.evaluate(test_mdp)
        if len(transformer_dqn.used) == test_mdp.num_nodes:
            transformer_dqn_trials.append(trans_dqn_total_dis)
            print(trans_dqn_routes)
        else:
            transformer_dqn_trials.append(200)

        gnn_dqn_total_dis, gnn_dqn_routes = gnn_dqn.evaluate(test_mdp)
        if len(gnn_dqn.used) == test_mdp.num_nodes:
            gnn_dqn_trials.append(gnn_dqn_total_dis)
            print(gnn_dqn_routes)
        else:
            gnn_dqn_trials.append(200)

        gnn_trans_dqn_total_dis, gnn_trans_dqn_routes = gnn_trans_dqn.evaluate(test_mdp)
        if len(gnn_trans_dqn.used) == test_mdp.num_nodes:
            gnn_trans_dqn_trials.append(gnn_trans_dqn_total_dis)
            print(gnn_trans_dqn_routes)
        else:
            gnn_trans_dqn_trials.append(200)

plt.plot(heuristic_trials, label="heuristic")
plt.plot(only_dqn_trials, label="only dqn")
plt.plot(transformer_dqn_trials, label="transformer dqn")
plt.plot(gnn_dqn_trials, label="gnn dqn")
plt.plot(gnn_trans_dqn_trials, label="gnn trans dqn")
plt.legend()
plt.savefig("node.png")
