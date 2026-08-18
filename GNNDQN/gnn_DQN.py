import torch.nn as nn
import torch
from copy import deepcopy
from itertools import chain
import GNNDQN.GNN_embedding as GNN
from DQN_parent import DQN


class GNNDQN(DQN):
    def __init__(self, input_dim_gnn, gnn_hidden_dim, gnn_output_dim, large_value, **parent_variebles):
        embedding_dim = gnn_output_dim - 3
        expected_input_dim = 3 * embedding_dim + 5
        if parent_variebles.get("input_dim") != expected_input_dim:
            raise ValueError(
                f"GNNDQN expects input_dim={expected_input_dim}, "
                f"got {parent_variebles.get('input_dim')}"
            )
        super().__init__(**parent_variebles)

        self.node_embedding = None
        self.gnn = GNN.SimpleGNN(input_dim=input_dim_gnn, hidden_dim=gnn_hidden_dim,
                                output_dim=gnn_output_dim, device=self.device,
                                large_value=large_value)
        self.gnn.to(device=self.device)
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

