import torch.nn as nn
import torch
from copy import deepcopy
from itertools import chain
import GNNDQN.GNN_embedding as GNN
from DQN_parent import DQN


class GNNDQN(DQN):
    def __init__(self, gnn_input_dim, gnn_hidden_dim, gnn_output_dim, large_value, **parent_variebles):
        super().__init__(**parent_variebles)

        self.gnn = GNN.SimpleGNN(node_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, 
                                output_dim=gnn_output_dim, edge_dim=1, device=self.device,
                                large_value=large_value)
        self.optimizer = torch.optim.Adam( # abetter optimizer what about the parameters 
            chain(self.explore_model.parameters(), 
            self.gnn.parameters()),
            lr=self.lr
        )

    def get_graph_statics(self, mdp, train):
        '''
        graph statics:
            is given from the transformer
        '''
        if not train:
            with torch.no_grad():
                self.node_embedding =  self.gnn(mdp)
        else:
            self.node_embedding = self.gnn(mdp)
        return self.node_embedding.mean(dim=0)
    
    def get_unused_nodes_statics(self, mdp, used):
        if self.node_embedding == None:
            print("error it must not self.node_embedding mudt bot be none")
        else:
            unused_embeddings = [self.node_embedding[i] for i in range(mdp.num_nodes) if i not in used]
            if not unused_embeddings:
                sum_unused_nodes = torch.zeros_like(self.node_embedding[0])
            else:
                sum_unused_nodes = torch.stack(unused_embeddings).mean(dim=0)

            return sum_unused_nodes
    
    def get_next_node_statics(self, mdp, route, next_node):
        if self.node_embedding == None:
            print("error it must not self.node_embedding mudt bot be none")
        else:
            dis = mdp.distance_matrix[route["current_node"]][next_node]
            dis = torch.tensor([dis], device=self.device, dtype=torch.float)
            
            return torch.cat([self.node_embedding[next_node], dis])
    





            

