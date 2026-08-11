import random

import yaml
import numpy as np
import torch

from MDP import MDP
from heuristic import ClarkeWrightSavings
import GNN_embedding as GNN
import graph_transformer as GT
from DQN import DQN
# from A2C import A2C


from plot import performance_comparison


with open("conf.yaml", "r") as file:
    config = yaml.safe_load(file)

# problem parametrs
SEED = config["problem"]["seed"]
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
DQN_MAX_NUM_NODES = config["DQN"]["max_num_nodes"]
DQN_MAX_NUM_CARS = config["DQN"]["max_num_cars"]
DQN_CARS_CAPACITY = config["DQN"]["cars_capacity"]

# A2C
A2C_HIDDEN_DIM = config["A2C"]["hidden_dim"]
A2C_OUTPUT_DIM = config["A2C"]["output_dim"]
A2C_RL = config["A2C"]["lr"]
A2C_DISCOUNT = config["A2C"]["discount"]
A2C_NUM_EPOCHES = config["A2C"]["num_epoches"]
A2C_ENTROPY_COEF = config["A2C"]["entropy_coef"]


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

    dqn_gnn_embedder = GNN.SimpleGNN(GNN_NODE_DIM, GNN_HiDDEN_DIM,
                                      GNN_OUTPUT_DIM, GNN_EDGE_DIM)
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
    }, "dqn_checkpoint.pt")
    print("Saved model parameters to dqn_checkpoint.pt")

    test_mdp = MDP(NUM_NODES, DEPOT_NUM, NUM_CARS, CARS_CAPACITY)
    test_mdp.fill_distance_matrix(MIN_DIS, MAX_DIS)
    test_mdp.fill_node_cap_matrix(MIN_CAP, MAX_CAP)

    print("Environment")
    print("Distance matrix")
    print(test_mdp.distance_matrix)
    print("node capacities")
    print(test_mdp.node_capacity)

    # heuristic result
    cws = ClarkeWrightSavings(test_mdp)
    feasible = cws.CWS_solve()

    print("The result of the hueristic approach")
    if not feasible:
        print("the heuristic did not find a solution")
    else:
        print(cws.list_routes)
    heuristic_dis = total_dis(cws.list_routes, test_mdp)
    print("heuristic distance:", heuristic_dis)

    test_node_feature = GNN.create_node_feature(test_mdp)
    test_edge_feature = GNN.create_edge_feature(test_mdp)
    test_gnn_input = GNN.create_input(test_node_feature, test_edge_feature, LARGE_VALUE)

    # ---------------- DQN ----------------

    dqn_routes = dqn.eval_model(test_mdp, test_gnn_input)
    dqn_dis = sum(route["total_distance"] for route in dqn_routes.values())
    for _, route in dqn_routes.items():
        print(route["path"] )
    print("dqn_result: ", dqn_dis)

    print()
    print("=== Summary (same problem instance for both) ===")
    print("heuristic:", heuristic_dis)
    print("dqn:      ", dqn_dis)

    # ---------------- A2C ----------------
    #
    # a2c_gnn_embedder = GNN.SimpleGNN(GNN_NODE_DIM, GNN_HiDDEN_DIM,
    #                                   GNN_OUTPUT_DIM, GNN_EDGE_DIM)
    # a2c_graph_transformer = GT.Encoder(NUM_LAYERS, NUM_HEAD, MODEL_DIM, FF_HIDDEN_DIM)
    # a2c = A2C(test_mdp, a2c_gnn_embedder, a2c_graph_transformer, test_gnn_input,
    #           A2C_NUM_EPOCHES, A2C_RL, A2C_HIDDEN_DIM, A2C_OUTPUT_DIM,
    #           A2C_DISCOUNT, entropy_coef=A2C_ENTROPY_COEF)
    # a2c.A2C_train()
    # a2c_routes = a2c.eval_model()
    # a2c_dis = sum(route["total_distance"] for route in a2c_routes.values())
    # print("a2c_result: ", a2c_dis)