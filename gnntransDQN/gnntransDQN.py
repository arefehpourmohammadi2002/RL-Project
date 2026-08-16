import torch.nn as nn
import torch
from copy import deepcopy
from itertools import chain

from gnntransDQN.graph_transformer import Encoder
from gnntransDQN.GNN_embedding import SimpleGNN
from DQN_parent import DQN
import replay_buffer as RB


class GNNTransDQN(DQN):
    def __init__(self, input_dim_gnn, gnn_hidden_dim, gnn_output_dim, num_layers, large_value, num_heads, model_dim, FF_hidden_dim, **parent_variebles):
        super().__init__(**parent_variebles)

        self.gnn = SimpleGNN(input_dim=input_dim_gnn, hidden_dim=gnn_hidden_dim, 
                        output_dim=gnn_output_dim, device=self.device,
                        large_value=large_value)
        
        self.transforemer = Encoder(num_layers=num_layers, num_heads=num_heads, 
                                    model_dim=model_dim, FF_hidden_dim=FF_hidden_dim)
        self.gnn.to(device=self.device)
        self.transforemer.to(self.device)
        self.optimizer = torch.optim.Adam( # abetter optimizer what about the parameters 
            chain(self.explore_model.parameters(), 
            self.transforemer.parameters(), 
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
                self.node_embedding = self.gnn(mdp)
                self.graph_embedding =  self.transforemer(self.node_embedding)
        else:
            self.node_embedding = self.gnn(mdp)
            self.graph_embedding =  self.transforemer(self.node_embedding)
        return self.graph_embedding
    
    def get_unused_nodes_statics(self, mdp, used):
        if self.node_embedding == None:
            print("error it must not self.node_embedding mudt bot be none")
        else:
            unused_embeddings = [self.node_embedding[i] for i in range(mdp.num_nodes) if i not in used]
            if not unused_embeddings:
                sum_unused_nodes = torch.zeros_like(self.node_embedding[0], device=self.device, dtype=torch.float)
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
    





            
