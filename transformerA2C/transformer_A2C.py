import torch
import torch.nn as nn
from A2C_parent import A2C
from transformerDQN.transformer import Encoder


class TransformerA2C(A2C):
    def __init__(self, num_layers, num_heads, model_dim, FF_hidden_dim, **parent_variables):
        parent_variables.setdefault("actor_input_dim", 3 * model_dim + 5)
        parent_variables.setdefault("critic_input_dim", 3 * model_dim + 5)
        super().__init__(**parent_variables)

        self.transforemer = Encoder(num_layers=num_layers, num_heads=num_heads, model_dim=model_dim, FF_hidden_dim=FF_hidden_dim, device=self.device)
        self.transforemer.to(self.device)
        self.node_embedding = None

    def get_graph_statics(self, mdp, train):
        if not train:
            with torch.no_grad():
                self.graph_embedding, self.node_embedding = self.transforemer(mdp)
        else:
            self.graph_embedding, self.node_embedding = self.transforemer(mdp)
        return self.graph_embedding

    def get_unused_nodes_statics(self, mdp, used):
        if self.node_embedding is None:
            return torch.zeros(3, device=self.device)
        unused = [self.node_embedding[i] for i in range(mdp.num_nodes) if i not in used]
        if not unused:
            return torch.zeros_like(self.node_embedding[0])
        return torch.stack(unused).mean(dim=0)

    def get_next_node_statics(self, mdp, route, next_node):
        if self.node_embedding is None:
            raise RuntimeError("node embeddings must be created first")
        distance = mdp.distance_matrix[route["current_node"]][next_node] / self.distance_scale(mdp)
        distance = torch.tensor([distance], device=self.device, dtype=torch.float32)
        return torch.cat([self.node_embedding[next_node], distance])
