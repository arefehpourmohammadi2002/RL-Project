import yaml

from MDP import MDP
from heuristic import ClarkeWrightSavings
import GNN_embedding  as GNN

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

# transformer parameters
GNN_NODE_DIM = config["graph_transformer"]["GNN_input_embedding"]["node_dim"]
GNN_HiDDEN_DIM = config["graph_transformer"]["GNN_input_embedding"]["hidden_dim"]
GNN_OUTPUT_DIM = config["graph_transformer"]["GNN_input_embedding"]["output_dime"]
GNN_EDGE_DIM = config["graph_transformer"]["GNN_input_embedding"]["edge_dim"]

if __name__ == "__main__":
    mdp = MDP(NUM_NODES, DEPOT_NUM, NUM_CARS, CARS_CAPACITY)
    mdp.fill_distance_matrix(MAX_DIS, MIN_DIS)
    mdp.fill_node_cap_matrix(MAX_CAP, MIN_CAP)

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

    ### GNN embedding for the input of the transformer ###
    ''' for now the node_feature is one number the capacity of that node'''

    gnn_input_embedder = GNN.SimpleGNN(GNN_NODE_DIM, GNN_HiDDEN_DIM, GNN_OUTPUT_DIM, GNN_EDGE_DIM)

    # by default the functions just return the cpacity matrix as the node feature and
    # distance matrix as the edge feature the point is but they are written this way to 
    # be possible to create other input forms for further enhancement 
    node_feature = GNN.create_node_feature(mdp)
    edge_feature = GNN.create_edge_feature(mdp)

    input = GNN.create_input(node_feature, edge_feature)

    embedded_input = gnn_input_embedder(input)
    print(embedded_input)