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
        embedding_dim = gnn_output_dim - 3
        if model_dim != embedding_dim:
            raise ValueError(
                f"model_dim must equal the GNN embedding dimension "
                f"({embedding_dim}), got {model_dim}"
            )
        expected_input_dim = 3 * embedding_dim + 5
        if parent_variebles.get("input_dim") != expected_input_dim:
            raise ValueError(
                f"GNNTransDQN expects input_dim={expected_input_dim}, "
                f"got {parent_variebles.get('input_dim')}"
            )
        super().__init__(**parent_variebles)

        self.graph_embedding = None
        self.node_embedding = None
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
        if self.node_embedding is None:
            raise RuntimeError("node embeddings must be created first")

        unused_embeddings = [
            self.node_embedding[i]
            for i in range(mdp.num_nodes)
            if i not in used
        ]
        if not unused_embeddings:
            return torch.zeros_like(self.node_embedding[0])
        return torch.stack(unused_embeddings).mean(dim=0)
    
    def get_next_node_statics(self, mdp, route, next_node):
        if self.node_embedding is None:
            raise RuntimeError("node embeddings must be created first")

        distance = (
            mdp.distance_matrix[route["current_node"]][next_node]
            / self.distance_scale(mdp)
        )
        distance = torch.tensor(
            [distance], device=self.device, dtype=torch.float32
        )
        return torch.cat([self.node_embedding[next_node], distance])
