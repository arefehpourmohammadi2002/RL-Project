import yaml
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from MDP import MDP
from heuristic import ClarkeWrightSavings
import GNN_embedding as GNN
import graph_transformer as GT
from DQN import DQN


from plot import performance_comparison


CHECKPOINT_PATH = "dqn_checkpoint_temp.pt"

with open("conf.yaml", "r") as file:
    config = yaml.safe_load(file)

# problem parameters
SEED = config["problem"]["seed"]
DEPOT_NUM = config["problem"]["depot_num"]
NUM_NODES = config["problem"]["num_nodes"]
NUM_CARS = config["problem"]["num_cars"]
CARS_CAPACITY = config["problem"]["cars_capacity"]
MAX_DIS = config["problem"]["max_dis"]
MIN_DIS = config["problem"]["min_dis"]
MAX_CAP = config["problem"]["max_cap"]
MIN_CAP = config["problem"]["min_cap"]

# GNN embedder parameters
GNN_NODE_DIM = config["GNN_input_embedding"]["node_dim"]
GNN_HiDDEN_DIM = config["GNN_input_embedding"]["hidden_dim"]
GNN_OUTPUT_DIM = config["GNN_input_embedding"]["output_dime"]
GNN_EDGE_DIM = config["GNN_input_embedding"]["edge_dim"]
LARGE_VALUE = config["GNN_input_embedding"]["large_value"]

# Graph transformer parameters
NUM_HEAD = config["graph_transformer"]["multi_head_attention"]["num_heads"]
MODEL_DIM = config["graph_transformer"]["multi_head_attention"]["model_dim"]
NUM_LAYERS = config["graph_transformer"]["num_layers"]
FF_HIDDEN_DIM = config["graph_transformer"]["ff_hidden_dim"]

# DQN parameters
DQN_OUTPUT_DIM = config["DQN"]["output_dim"]
DQN_HIDDEN_DIM = config["DQN"]["hidden_dim"]
DQN_NUM_EPOCHES = config["DQN"]["num_epoches"]
DQN_MAX_NUM_NODES = config["DQN"]["max_num_nodes"]
DQN_MAX_NUM_CARS = config["DQN"]["max_num_cars"]
DQN_CARS_CAPACITY = config["DQN"]["cars_capacity"]
EPSILON = config["DQN"]["epsilon"]
EPSILON_DECAY = config["DQN"]["epsilon_decay"]
DQN_RL = config["DQN"]["rl"]
EXP_UP_STEP = config["DQN"]["explore_model_update_step"]
TRGET_UP_STEP = config["DQN"]["target_model_update_step"]
REPLAY_BUF_CAP = config["DQN"]["replay_buff_cap"]
REPLAY_BUF_FIRST_SIZE = config["DQN"]["replay_buffer_first_size"]
DQN_DISCOUNT = config["DQN"]["discount"]
BATCH_SIZE = config["DQN"]["batch_size"]
DQN_LR_DECAY = config["DQN"]["rl_decay"]
DQN_MIN_LR = config["DQN"]["min_rl"]

# Grid test
GRID_MIN_NODES = config["grid_test"]["min_nodes"]
GRID_MAX_NODES = config["grid_test"]["max_nodes"]
GRID_MIN_CARS = config["grid_test"]["min_cars"]
GRID_MAX_CARS = config["grid_test"]["max_cars"]
GRID_TRIALS = config["grid_test"]["trials_per_cell"]
GRID_BASE_FILENAME = config["grid_test"]["base_filename"]


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


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    init_mdp = MDP(NUM_NODES, DEPOT_NUM, NUM_CARS, CARS_CAPACITY)
    init_mdp.fill_distance_matrix(MIN_DIS, MAX_DIS)
    init_mdp.fill_node_cap_matrix(MIN_CAP, MAX_CAP)

    init_node_feature = GNN.create_node_feature(init_mdp)
    init_edge_feature = GNN.create_edge_feature(init_mdp)
    init_gnn_input = GNN.create_input(init_node_feature, init_edge_feature, LARGE_VALUE)

    dqn_gnn_embedder = GNN.SimpleGNN(GNN_NODE_DIM, GNN_HiDDEN_DIM, GNN_OUTPUT_DIM, GNN_EDGE_DIM)
    dqn_graph_transformer = GT.Encoder(NUM_LAYERS, NUM_HEAD, MODEL_DIM, FF_HIDDEN_DIM)

    dqn = DQN(
        mdp=init_mdp,
        gnn_model=dqn_gnn_embedder,
        transformer_model=dqn_graph_transformer,
        gnn_input=init_gnn_input,
        num_epoches=DQN_NUM_EPOCHES,
        explore_model_update_step=EXP_UP_STEP,
        target_model_update_step=TRGET_UP_STEP,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        lr=DQN_RL,
        lr_decay=DQN_LR_DECAY,
        min_lr=DQN_MIN_LR,
        hiden_dim=DQN_HIDDEN_DIM,
        output_dim=DQN_OUTPUT_DIM,
        replay_buff_cap=REPLAY_BUF_CAP,
        replay_buffer_first_size=REPLAY_BUF_FIRST_SIZE,
        discount=DQN_DISCOUNT,
        batch_size=BATCH_SIZE,
        min_num_nodes=5,
        max_num_nodes=DQN_MAX_NUM_NODES,
        min_num_cars=1,
        max_num_cars=DQN_MAX_NUM_CARS,
        cars_capacity=DQN_CARS_CAPACITY,
        min_dis=MIN_DIS,
        max_dis=MAX_DIS,
        min_node_cap=MIN_CAP,
        max_node_cap=MAX_CAP,
        large_value=LARGE_VALUE,
        depot_num=DEPOT_NUM,
    )

    dqn.DQN_train()

    torch.save({
        "gnn_model_state": dqn.gnn_model.state_dict(),
        "transformer_model_state": dqn.transformer_model.state_dict(),
        "explore_model_state": dqn.explore_model.state_dict(),
        "target_model_state": dqn.target_model.state_dict(),
    }, CHECKPOINT_PATH)
    # window = 100
    # smooth = np.convolve(dqn.loss_list, np.ones(window)/window, mode='valid')
    # plt.plot(dqn.loss_list, alpha=0.4, color='gray')
    # plt.plot(range(window-1, len(dqn.loss_list)), smooth, color='red', linewidth=3)
    # plt.savefig('loss.png')


    # ---- RL methods vs CW savings heuristic ----
    node_values = list(range(GRID_MIN_NODES, GRID_MAX_NODES + 1))
    car_values = list(range(GRID_MIN_CARS, GRID_MAX_CARS + 1))


    heuristic_avg_dis = np.zeros((len(node_values), len(car_values)))
    dqn_avg_dis = np.zeros((len(node_values), len(car_values)))
    heuristic_feasible_rate = np.zeros((len(node_values), len(car_values)))
    dqn_feasible_rate = np.zeros((len(node_values), len(car_values)))

    for i, n_nodes in enumerate(node_values):
        for j, n_cars in enumerate(car_values):

            heuristic_trials = []
            dqn_trials = []

            for trial in range(GRID_TRIALS):
                test_mdp = MDP(n_nodes, DEPOT_NUM, n_cars, CARS_CAPACITY)
                test_mdp.fill_distance_matrix(MIN_DIS, MAX_DIS)
                test_mdp.fill_node_cap_matrix(MIN_CAP, MAX_CAP)

                cws = ClarkeWrightSavings(test_mdp)
                feasible = cws.CWS_solve()
                if feasible:
                    heuristic_trial_dis = total_dis(cws.list_routes, test_mdp)
                    heuristic_trials.append(heuristic_trial_dis)

                test_node_feature = GNN.create_node_feature(test_mdp)
                test_edge_feature = GNN.create_edge_feature(test_mdp)
                test_gnn_input = GNN.create_input(test_node_feature, test_edge_feature, LARGE_VALUE)

                dqn_routes = dqn.eval_model(test_mdp, test_gnn_input)

                unvisited_count = (test_mdp.num_nodes - 1) - len(dqn.used)
                if unvisited_count == 0:
                    dqn_trial_dis = sum(r["total_distance"] for r in dqn_routes.values())
                    dqn_trials.append(dqn_trial_dis)

            heuristic_feasible_rate[i, j] = len(heuristic_trials) / GRID_TRIALS
            dqn_feasible_rate[i, j] = len(dqn_trials) / GRID_TRIALS

            if heuristic_trials:
                heuristic_avg_dis[i, j] = np.mean(heuristic_trials)

            if dqn_trials:
                dqn_avg_dis[i, j] = np.mean(dqn_trials)

            # print(f"nodes={n_nodes} cars={n_cars} | "
            #       f"heuristic avg={heuristic_avg_dis[i, j]:10.3f} "
            #       f"(feasible {heuristic_feasible_rate[i, j]:5.0%}) | "
            #       f"dqn avg={dqn_avg_dis[i, j]:10.3f} "
            #       f"(feasible {dqn_feasible_rate[i, j]:5.0%})")

    print()
    print("=== Grid summary (rows=num_nodes, cols=num_cars) ===")
    print("Heuristic average distance (feasible trials only):")
    print(heuristic_avg_dis)
    print("Heuristic feasibility rate:")
    print(heuristic_feasible_rate)
    print("DQN average distance (feasible trials only):")
    print(dqn_avg_dis)
    print("DQN feasibility rate:")
    print(dqn_feasible_rate)


    with np.errstate(divide="ignore", invalid="ignore"):
        heuristic_penalized = np.where(heuristic_feasible_rate > 0,
                                        heuristic_avg_dis / heuristic_feasible_rate,
                                        np.nan)
        dqn_penalized = np.where(dqn_feasible_rate > 0,
                                  dqn_avg_dis / dqn_feasible_rate,
                                  np.nan)

    finite_values = np.concatenate([
        heuristic_penalized[~np.isnan(heuristic_penalized)],
        dqn_penalized[~np.isnan(dqn_penalized)],
    ])
    if finite_values.size > 0:
        worst_case = finite_values.max() * 1.2
    else:
        worst_case = 1.0  
    heuristic_penalized = np.where(np.isnan(heuristic_penalized), worst_case, heuristic_penalized)
    dqn_penalized = np.where(np.isnan(dqn_penalized), worst_case, dqn_penalized)

    print()
    print(f"=== Feasibility-penalized distance (avg_dis / feasible_rate, "
          f"0%-feasible cells set to {worst_case:.3f}) ===")
    print("Heuristic:")
    print(heuristic_penalized)
    print("DQN:")
    print(dqn_penalized)



    performance_comparison(
        heuristic_penalized,
        dqn_penalized,
        start_nodes=GRID_MIN_NODES,
        base_filename=GRID_BASE_FILENAME,
    )
    print(f"Saved distance plots: {GRID_BASE_FILENAME}_3d.jpg, "
            f"{GRID_BASE_FILENAME}_by_nodes.jpg, {GRID_BASE_FILENAME}_by_cars.jpg")

    rate_base_filename = f"{GRID_BASE_FILENAME}_feasibility_rate"
    performance_comparison(
        heuristic_feasible_rate,
        dqn_feasible_rate,
        start_nodes=GRID_MIN_NODES,
        base_filename=rate_base_filename,
    )
    print(f"Saved feasibility-rate plots: {rate_base_filename}_3d.jpg, "
            f"{rate_base_filename}_by_nodes.jpg, {rate_base_filename}_by_cars.jpg")