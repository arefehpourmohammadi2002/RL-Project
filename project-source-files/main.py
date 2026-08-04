import yaml

from MDP import MDP
# from heuristic import ClarkeWrightSavings  
import GNN_embedding as GNN
import graph_transformer as GT
from DQN import DQN

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

# Graph Transformer parameters
NUM_HEAD = config["graph_transformer"]["multi_head_attention"]["num_heads"]
MODEL_DIM = config["graph_transformer"]["multi_head_attention"]["model_dim"]
NUM_LAYERS = config["graph_transformer"]["num_layers"]
FF_HIDDEN_DIM = config["graph_transformer"]["ff_hidden_dim"]

# DQN
EPSILON = config["DQN"]["epsilon"]
EPSILON_DECAY = config["DQN"]["epsilon_decay"]
RL = config["DQN"]["rl"]
DQN_OUTPUT_DIM = config["DQN"]["output_dim"]
DQN_HIDDEN_DIM = config["DQN"]["hidden_dim"]
NUM_EPOCHES = config["DQN"]["num_epoches"]
EXP_UP_STEP = config["DQN"]["explore_model_update_step"]
TRGET_UP_STEP = config["DQN"]["target_model_update_step"]
REPLAY_BUF_CAP = config["DQN"]["replay_buff_cap"]
REPLAY_BUF_FIRST_SIZE = config["DQN"]["replay_buffer_first_size"]
DISCOUNT = config["DQN"]["discount"]
BATCH_SIZE = config["DQN"]["batch_size"]

if __name__ == "__main__":
    mdp = MDP(NUM_NODES, DEPOT_NUM, NUM_CARS, CARS_CAPACITY)

    mdp.fill_distance_matrix(MIN_DIS, MAX_DIS)
    mdp.fill_node_cap_matrix(MIN_CAP, MAX_CAP)

    print("Environment")
    print("Distance matrix")
    print(mdp.distance_matrix)
    print("node capacities")
    print(mdp.node_capacity)

    ### heuristic result ###
    # cws = ClarkeWrightSavings(mdp)
    # feasible = cws.CWS_solve()

    # print("The result of the hueristic approach")
    # if not feasible:
    #     print("the heuristic did not find a solution")
    # else:
    #     print(cws.list_routes)

    ### GNN 
    node_feature = GNN.create_node_feature(mdp)
    edge_feature = GNN.create_edge_feature(mdp)
    gnn_input = GNN.create_input(node_feature, edge_feature)

    gnn_input_embedder = GNN.SimpleGNN(GNN_NODE_DIM, GNN_HiDDEN_DIM,
                                        GNN_OUTPUT_DIM, GNN_EDGE_DIM)
    graph_transformer = GT.Encoder(NUM_LAYERS, NUM_HEAD, MODEL_DIM, FF_HIDDEN_DIM)

    ### DQN
    dqn = DQN(mdp, gnn_input_embedder, graph_transformer, gnn_input,
              NUM_EPOCHES, EXP_UP_STEP, TRGET_UP_STEP,
              EPSILON, EPSILON_DECAY, RL, DQN_HIDDEN_DIM, DQN_OUTPUT_DIM,
              REPLAY_BUF_CAP, REPLAY_BUF_FIRST_SIZE, discount=DISCOUNT,
              batch_size=BATCH_SIZE)
    dqn.DQN_train()