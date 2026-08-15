import torch.nn as nn
import torch
from copy import deepcopy

import GNN_embedding as GNN
from graph_transformer import Encoder
from ..DQN_parent import DQN

class DQN:
    def __init__(self, lr, large_value, **parent_variebles):
        super().__init__(**parent_variebles)

        self.gnn = GNN.SimpleGNN()
        self.transforemer = Encoder()
        self.optimizer = torch.optim.Adam( # abetter optimizer what about the parameters 
            self.explore_model.parameters(), 
            self.gnn.parameters(),
            self.transforemer.parameters(),
            lr=lr
        )

        self.large_value = large_value

    def get_graph_statics(self, mdp, train):
        '''
        graph statics:
            is given from the GNN
        '''
        node_feature = GNN.create_node_feature(mdp=mdp)
        edge_feature = GNN.create_edge_feature(mdp=mdp)
        input = GNN.create_input(node_feature, edge_feature, self.large_value)

        if not train:
            with torch.no_grad():
                gnn_output = self.gnn(input)
                return self.transforemer(gnn_output)
        
        gnn_output = self.gnn(input)
        return self.transforemer(gnn_output)
