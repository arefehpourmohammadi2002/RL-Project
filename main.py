import matplotlib.pyplot as plt

from onlyDQN.only_DQN import OnlyDQN
from transformerDQN.transformer_DQN import TransformerDQN
from GNNDQN.gnn_DQN import GNNDQN
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

output_dim=1
hidden_dim=32
lr=1e-3
target_update_counter=10
explore_update_counter=2
discount=0.95
epsilon=0.3
batch_size=20
RB_capacity=500
num_first_samples=32
num_epoches=10
max_num_nodes=10
min_num_nodes=5
depot_num=0
max_num_cars=5
min_num_cars=3
cars_capacity=15.0
min_distance=1.0
max_distance=10.0
min_node_dem=1.0
max_node_dem=4.0

gnn_dqn = GNNDQN(
        gnn_input_dim=7,
        gnn_hidden_dim=1,
        gnn_output_dim=5,
        large_value=100,
        input_dim=20,
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
        input_dim=15,
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
        input_dim=20,
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


gnn_dqn.DQN_train()
only_dqn.DQN_train()
transformer_dqn.DQN_train()

heuristic_trials = []
only_dqn_trials = []
transformer_dqn_trials = []
gnn_dqn_trials = []

for n_nodes in range(5, 10):
    for n_cars in range(1, 4):
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
        
        dqn_total_dis, dqn_routes = only_dqn.evaluate(test_mdp)
        if len(only_dqn.used) == test_mdp.num_nodes:
            only_dqn_trials.append(dqn_total_dis)
            print(dqn_routes)

        trans_dqn_total_dis, trans_dqn_routes = transformer_dqn.evaluate(test_mdp)
        if len(only_dqn.used) == test_mdp.num_nodes:
            transformer_dqn_trials.append(trans_dqn_total_dis)
            print(trans_dqn_routes)

        gnn_dqn_total_dis, gnn_dqn_routes = transformer_dqn.evaluate(test_mdp)
        if len(only_dqn.used) == test_mdp.num_nodes:
            gnn_dqn_trials.append(gnn_dqn_total_dis)
            print(gnn_dqn_routes)
        

plt.plot(heuristic_trials, label="heuristic")
plt.plot(only_dqn_trials, label="only dqn")
plt.plot(transformer_dqn_trials, label="transformer dqn")
plt.plot(gnn_dqn_trials, label="gnn dqn")
plt.legend()
plt.savefig("node.png")

