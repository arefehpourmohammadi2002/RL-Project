import yaml
import numpy as np
import random
import torch

from MDP import MDP
from heuristic import ClarkeWrightSavings
import GNN_embedding as GNN
import graph_transformer as GT
from DQN import DQN
from A2C import A2C

try:
    from plot import performance_comparison
except ImportError:
    performance_comparison = None

with open("conf.yaml", "r") as file:
    config = yaml.safe_load(file)

# problem parametrs
NUM_NODES = config["problem"]["num_nodes"]
DEPOT_NUM = config["problem"]["depot_num"]
NUM_CARS = config["problem"]["num_cars"]
CARS_CAPACITY = config["problem"]["cars_capacity"]
MAX_DIS = config["problem"]["max_dis"]
MIN_DIS = config["problem"]["min_dis"]
MAX_CAP = config["problem"]["max_cap"]
MIN_CAP = config["problem"]["min_cap"]

# GNN_embeder parameters
GNN_NODE_DIM = config["GNN_input_embedding"]["node_dim"]
GNN_HiDDEN_DIM = config["GNN_input_embedding"]["hidden_dim"]
GNN_OUTPUT_DIM = config["GNN_input_embedding"]["output_dime"]
GNN_EDGE_DIM = config["GNN_input_embedding"]["edge_dim"]
LARGE_VALUE = config["GNN_input_embedding"]["large_value"]

# Graph Transformer parameters
NUM_HEAD = config["graph_transformer"]["multi_head_attention"]["num_heads"]
MODEL_DIM = config["graph_transformer"]["multi_head_attention"]["model_dim"]
NUM_LAYERS = config["graph_transformer"]["num_layers"]
FF_HIDDEN_DIM = config["graph_transformer"]["ff_hidden_dim"]

# DQN
EPSILON = config["DQN"]["epsilon"]
EPSILON_DECAY = config["DQN"]["epsilon_decay"]
DQN_RL = config["DQN"]["rl"]
DQN_OUTPUT_DIM = config["DQN"]["output_dim"]
DQN_HIDDEN_DIM = config["DQN"]["hidden_dim"]
DQN_NUM_EPOCHES = config["DQN"]["num_epoches"]
EXP_UP_STEP = config["DQN"]["explore_model_update_step"]
TRGET_UP_STEP = config["DQN"]["target_model_update_step"]
REPLAY_BUF_CAP = config["DQN"]["replay_buff_cap"]
REPLAY_BUF_FIRST_SIZE = config["DQN"]["replay_buffer_first_size"]
DQN_DISCOUNT = config["DQN"]["discount"]
BATCH_SIZE = config["DQN"]["batch_size"]

# A2C
A2C_HIDDEN_DIM = config["A2C"]["hidden_dim"]
A2C_OUTPUT_DIM = config["A2C"]["output_dim"]
A2C_RL = config["A2C"]["lr"]
A2C_DISCOUNT = config["A2C"]["discount"]
A2C_NUM_EPOCHES = config["A2C"]["num_epoches"]
A2C_ENTROPY_COEF = config["A2C"]["entropy_coef"]

SEED = 43
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

    huerisric_performance = np.full((NUM_NODES + 1, NUM_CARS + 1), float(1.0))
    DQN_performance = np.full((NUM_NODES + 1, NUM_CARS + 1), float(1.0))

    mdp = MDP(NUM_NODES, DEPOT_NUM, NUM_CARS, CARS_CAPACITY)
    mdp.fill_distance_matrix(MIN_DIS, MAX_DIS)
    mdp.fill_node_cap_matrix(MIN_CAP, MAX_CAP)

    print("Environment")
    print("Distance matrix")
    print(mdp.distance_matrix)
    print("node capacities")
    print(mdp.node_capacity)

    # heuristic result
    cws = ClarkeWrightSavings(mdp)
    feasible = cws.CWS_solve()

    print("The result of the hueristic approach")
    if not feasible:
        print("the heuristic did not find a solution")
    else:
        print(cws.list_routes)
    heuristic_dis = total_dis(cws.list_routes, mdp)
    print("heuristic distance:", heuristic_dis)
    # huerisric_performance[num_nodes][num_cars] = heuristic_dis

    # Raw GNN input features -- built once from mdp, reused by both DQN and A2C.
    node_feature = GNN.create_node_feature(mdp)
    edge_feature = GNN.create_edge_feature(mdp)
    gnn_input = GNN.create_input(node_feature, edge_feature, LARGE_VALUE)

    # ---------------- DQN ----------------
    dqn_gnn_embedder = GNN.SimpleGNN(GNN_NODE_DIM, GNN_HiDDEN_DIM,
                                      GNN_OUTPUT_DIM, GNN_EDGE_DIM)
    dqn_graph_transformer = GT.Encoder(NUM_LAYERS, NUM_HEAD, MODEL_DIM, FF_HIDDEN_DIM)

    dqn = DQN(mdp, dqn_gnn_embedder, dqn_graph_transformer, gnn_input,
              DQN_NUM_EPOCHES, EXP_UP_STEP, TRGET_UP_STEP,
              EPSILON, EPSILON_DECAY, DQN_RL, DQN_HIDDEN_DIM, DQN_OUTPUT_DIM,
              REPLAY_BUF_CAP, REPLAY_BUF_FIRST_SIZE, discount=DQN_DISCOUNT,
              batch_size=BATCH_SIZE)
    dqn.DQN_train()
    dqn_routes = dqn.eval_model()
    dqn_dis = sum(route["total_distance"] for route in dqn_routes.values())
    # DQN_performance[num_nodes][num_cars] = dqn_dis
    print("dqn_result: ", dqn_dis)

    # ---------------- A2C ----------------
    a2c_gnn_embedder = GNN.SimpleGNN(GNN_NODE_DIM, GNN_HiDDEN_DIM,
                                      GNN_OUTPUT_DIM, GNN_EDGE_DIM)
    a2c_graph_transformer = GT.Encoder(NUM_LAYERS, NUM_HEAD, MODEL_DIM, FF_HIDDEN_DIM)

    a2c = A2C(mdp, a2c_gnn_embedder, a2c_graph_transformer, gnn_input,
              A2C_NUM_EPOCHES, A2C_RL, A2C_HIDDEN_DIM, A2C_OUTPUT_DIM,
              A2C_DISCOUNT, entropy_coef=A2C_ENTROPY_COEF)
    a2c.A2C_train()
    a2c_routes = a2c.eval_model()
    a2c_dis = sum(route["total_distance"] for route in a2c_routes.values())
    print("a2c_result: ", a2c_dis)

    print()
    print("=== Summary (same problem instance for all three) ===")
    print("heuristic:", heuristic_dis)
    print("dqn:      ", dqn_dis)
    print("a2c:      ", a2c_dis)

    if performance_comparison is not None:
        pass  # performance_comparison(huerisric_performance, DQN_performance)