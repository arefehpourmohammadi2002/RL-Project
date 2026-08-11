import yaml
import numpy as np
import torch

from MDP import MDP
from heuristic import ClarkeWrightSavings
import GNN_embedding as GNN
import graph_transformer as GT
from DQN import DQN

try:
    from plot import performance_comparison
except ImportError:
    performance_comparison = None

CHECKPOINT_PATH = "dqn_checkpoint.pt"

with open("conf.yaml", "r") as file:
    config = yaml.safe_load(file)

# problem parameters
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
    print(f"Saved model parameters to {CHECKPOINT_PATH}")

    # ---- grid sweep vs CW savings heuristic ----
    node_values = list(range(GRID_MIN_NODES, GRID_MAX_NODES + 1))
    car_values = list(range(GRID_MIN_CARS, GRID_MAX_CARS + 1))

    heuristic_perf = np.zeros((len(node_values), len(car_values)))
    dqn_perf = np.zeros((len(node_values), len(car_values)))

    # Feasibility-weighted metric, one value per num_nodes:
    #   (average distance over only the FEASIBLE trials for this node count,
    #    across every num_cars value) * (feasible trial count / total trial count)
    heuristic_weighted_by_nodes = np.zeros(len(node_values))
    dqn_weighted_by_nodes = np.zeros(len(node_values))

    for i, n_nodes in enumerate(node_values):

        # Accumulated across every num_cars value and every trial for this
        # num_nodes row -- this is what the feasibility-weighted metric below
        # is computed from.
        row_heuristic_feasible_dis = []
        row_heuristic_total_trials = 0
        row_dqn_feasible_dis = []
        row_dqn_total_trials = 0

        for j, n_cars in enumerate(car_values):

            heuristic_trials = []
            dqn_trials = []

            for trial in range(GRID_TRIALS):
                test_mdp = MDP(n_nodes, DEPOT_NUM, n_cars, CARS_CAPACITY)
                test_mdp.fill_distance_matrix(MIN_DIS, MAX_DIS)
                test_mdp.fill_node_cap_matrix(MIN_CAP, MAX_CAP)

                cws = ClarkeWrightSavings(test_mdp)
                feasible = cws.CWS_solve()
                if not feasible:
                    print(f"  [nodes={n_nodes} cars={n_cars} trial={trial}] "
                          f"heuristic needed more than {n_cars} vehicle(s); "
                          f"distance still computed from its routes")
                heuristic_dis = total_dis(cws.list_routes, test_mdp)
                heuristic_trials.append(heuristic_dis)

                row_heuristic_total_trials += 1
                if feasible:
                    row_heuristic_feasible_dis.append(heuristic_dis)

                test_node_feature = GNN.create_node_feature(test_mdp)
                test_edge_feature = GNN.create_edge_feature(test_mdp)
                test_gnn_input = GNN.create_input(test_node_feature, test_edge_feature, LARGE_VALUE)

                dqn_routes = dqn.eval_model(test_mdp, test_gnn_input)

                # BUGFIX: `if not dqn.used` only catches the case where ZERO
                # nodes were ever visited -- it silently misses the much more
                # common case of PARTIAL coverage (some nodes left unvisited).
                # Compare the actual visited count against how many non-depot
                # nodes exist instead. Mirrors the heuristic's own
                # infeasibility handling above: warn, but still use the real
                # (finite) distance -- injecting -inf into dqn_trials would
                # poison np.mean() for the whole cell and break the plot's
                # z-axis scaling, while the heuristic side only ever warns.
                unvisited_count = (test_mdp.num_nodes - 1) - len(dqn.used)
                dqn_feasible = unvisited_count == 0
                if not dqn_feasible:
                    print(f"  [nodes={n_nodes} cars={n_cars} trial={trial}] "
                          f"DQN left {unvisited_count} node(s) unvisited")
                dqn_dis = sum(r["total_distance"] for r in dqn_routes.values())
                dqn_trials.append(dqn_dis)

                row_dqn_total_trials += 1
                if dqn_feasible:
                    row_dqn_feasible_dis.append(dqn_dis)

            heuristic_mean = np.mean(heuristic_trials)
            heuristic_perf[i, j] = 0.0 if np.isnan(heuristic_mean) else heuristic_mean

            dqn_mean = np.mean(dqn_trials)
            dqn_perf[i, j] = 0.0 if np.isnan(dqn_mean) else dqn_mean

            print(f"nodes={n_nodes:>3} cars={n_cars:>3} | "
                  f"heuristic={heuristic_perf[i, j]:10.3f} | dqn={dqn_perf[i, j]:10.3f}")

        heuristic_ratio = (len(row_heuristic_feasible_dis) / row_heuristic_total_trials
                            if row_heuristic_total_trials > 0 else 0.0)
        heuristic_avg_feasible = (np.mean(row_heuristic_feasible_dis)
                                   if row_heuristic_feasible_dis else 0.0)
        heuristic_weighted_by_nodes[i] = heuristic_avg_feasible * heuristic_ratio

        dqn_ratio = (len(row_dqn_feasible_dis) / row_dqn_total_trials
                     if row_dqn_total_trials > 0 else 0.0)
        dqn_avg_feasible = np.mean(row_dqn_feasible_dis) if row_dqn_feasible_dis else 0.0
        dqn_weighted_by_nodes[i] = dqn_avg_feasible * dqn_ratio

    print()
    print("=== Grid summary (rows=num_nodes, cols=num_cars) ===")
    print("Heuristic distances:")
    print(heuristic_perf)
    print("DQN distances:")
    print(dqn_perf)

    if performance_comparison is None:
        print("plot.py not found - skipping plot generation.")
    else:
        performance_comparison(
            heuristic_perf,
            dqn_perf,
            start_nodes=GRID_MIN_NODES,
            base_filename=GRID_BASE_FILENAME,
        )
        print(f"Saved plots: {GRID_BASE_FILENAME}_3d.jpg, "
              f"{GRID_BASE_FILENAME}_by_nodes.jpg, {GRID_BASE_FILENAME}_by_cars.jpg")

    print()
    print("=== Feasibility-weighted distance by num_nodes ===")
    print("(average distance over feasible trials only, times the feasible-trial ratio,")
    print(" pooled across every num_cars value for that num_nodes)")
    print("Heuristic:", heuristic_weighted_by_nodes)
    print("DQN:      ", dqn_weighted_by_nodes)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(node_values, heuristic_weighted_by_nodes, label="Heuristic",
                color="navy", marker="o")
        ax.plot(node_values, dqn_weighted_by_nodes, label="DQN",
                color="darkred", marker="s")
        ax.set_xlabel("Number of Nodes", fontsize=11)
        ax.set_ylabel("Feasibility-Weighted Average Distance", fontsize=11)
        ax.set_title("Feasibility-Weighted Distance by Number of Nodes", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()

        weighted_filename = f"{GRID_BASE_FILENAME}_weighted_by_nodes.jpg"
        plt.savefig(weighted_filename, format="jpg", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {weighted_filename}")
    except ImportError:
        print("matplotlib not found - skipping feasibility-weighted plot.")