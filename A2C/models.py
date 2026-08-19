from itertools import chain

import numpy as np
import torch

from A2C.a2c_parent import A2C
from GNNDQN.GNN_embedding import SimpleGNN
from gnntransDQN.graph_transformer import Encoder as GraphEncoder
from transformerDQN.transformer import Encoder as PairEncoder


class OnlyA2C(A2C):
    INPUT_DIM = 21
    STATE_DIM = 12
    ACTION_DIM = 9

    def __init__(self, **parent_variables):
        if parent_variables.get("input_dim") != self.INPUT_DIM:
            raise ValueError(f"OnlyA2C expects input_dim={self.INPUT_DIM}")
        hidden = parent_variables["hidden_dim"]
        super().__init__(
            state_net_input_dim=self.STATE_DIM,
            state_net_hidden_dim=hidden,
            state_net_output_dim=hidden,
            action_net_input_dim=self.ACTION_DIM,
            action_net_hidden_dim=hidden,
            action_net_output_dim=hidden,
            **parent_variables,
        )
        self.optimizer = torch.optim.AdamW(self.parameters_for_optimizer(), lr=self.lr)

    def parameters_for_optimizer(self):
        return chain(self.actor.parameters(), self.critic.parameters(),
                     self.action_net.parameters(), self.state_net.parameters())

    def get_graph_statics(self, mdp, train):
        del train
        scale = self.distance_scale(mdp)
        distances = mdp.distance_matrix[~np.eye(mdp.num_nodes, dtype=bool)]
        total_demand = float(np.sum(mdp.node_demand))
        values = [
            mdp.num_nodes / max(self.max_num_nodes, 1),
            mdp.num_cars / max(self.max_num_cars, 1),
            float(np.mean(distances)) / scale,
            float(np.std(distances)) / scale,
            mdp.cars_capacity / max(total_demand, 1e-6),
            mdp.node_dem_ave / max(mdp.cars_capacity, 1e-6),
            total_demand / max(mdp.num_cars * mdp.cars_capacity, 1e-6),
        ]
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def get_unused_nodes_statics(self, mdp, used):
        unused = self.get_unused_nodes(mdp, used)
        demand = float(sum(mdp.node_demand[i] for i in unused))
        return torch.tensor([
            len(unused) / max(mdp.num_nodes - 1, 1),
            demand / len(unused) / mdp.cars_capacity if unused else 0.0,
            demand / max(mdp.num_cars * mdp.cars_capacity, 1e-6),
        ], device=self.device, dtype=torch.float32)

    def get_next_node_statics(self, mdp, route, next_node):
        scale = self.distance_scale(mdp)
        current, depot = route["current_node"], mdp.depot_num
        distance = mdp.distance_matrix[current][next_node]
        depot_distance = mdp.distance_matrix[depot][next_node]
        marginal = distance + depot_distance - mdp.distance_matrix[current][depot]
        saving = mdp.distance_matrix[current][depot] + depot_distance - distance
        return torch.tensor([
            float(np.sum(mdp.distance_matrix[:, next_node])) /
            max((mdp.num_nodes - 1) * scale, scale),
            mdp.node_demand[next_node] / max(mdp.cars_capacity, 1e-6),
            distance / scale, depot_distance / scale, marginal / scale, saving / scale,
            (mdp.cars_capacity - route["capacity"] - mdp.node_demand[next_node]) /
            max(mdp.cars_capacity, 1e-6),
        ], device=self.device, dtype=torch.float32)


class EmbeddingA2C(A2C):
    """Shared feature methods for graph-embedding A2C variants."""

    def setup_embedding_parent(self, embedding_dim, parent_variables):
        expected = 3 * embedding_dim + 5
        if parent_variables.get("input_dim") != expected:
            raise ValueError(f"expected input_dim={expected}")
        hidden = parent_variables["hidden_dim"]
        super().__init__(
            state_net_input_dim=2 * embedding_dim + 2,
            state_net_hidden_dim=hidden,
            state_net_output_dim=hidden,
            action_net_input_dim=embedding_dim + 3,
            action_net_hidden_dim=hidden,
            action_net_output_dim=hidden,
            **parent_variables,
        )
        self.graph_embedding = None
        self.node_embedding = None

    def get_unused_nodes_statics(self, mdp, used):
        unused = [self.node_embedding[i] for i in range(mdp.num_nodes) if i not in used]
        return torch.stack(unused).mean(0) if unused else torch.zeros_like(self.node_embedding[0])

    def get_next_node_statics(self, mdp, route, next_node):
        distance = torch.tensor([
            mdp.distance_matrix[route["current_node"]][next_node] / self.distance_scale(mdp)
        ], device=self.device, dtype=torch.float32)
        return torch.cat([self.node_embedding[next_node], distance])

    def embedding_parameters(self):
        return []

    def finish_setup(self):
        parameters = chain(self.actor.parameters(), self.critic.parameters(),
                           self.action_net.parameters(), self.state_net.parameters(),
                           self.embedding_parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr)


class GNNA2C(EmbeddingA2C):
    def __init__(self, input_dim_gnn, gnn_hidden_dim, gnn_output_dim,
                 large_value, **parent_variables):
        embedding_dim = gnn_output_dim - 3
        self.setup_embedding_parent(embedding_dim, parent_variables)
        self.gnn = SimpleGNN(input_dim_gnn, gnn_hidden_dim, gnn_output_dim,
                             self.device, large_value).to(self.device)
        self.finish_setup()

    def embedding_parameters(self):
        return self.gnn.parameters()

    def get_graph_statics(self, mdp, train):
        with torch.set_grad_enabled(train):
            self.node_embedding = self.gnn(mdp)
        return self.node_embedding.mean(0)

    def checkpoint(self):
        result = super().checkpoint()
        result["gnn"] = self.gnn.state_dict()
        return result

    def load_checkpoint(self, checkpoint):
        super().load_checkpoint(checkpoint)
        self.gnn.load_state_dict(checkpoint["gnn"])


class TransformerA2C(EmbeddingA2C):
    def __init__(self, num_layers, num_heads, model_dim, FF_hidden_dim,
                 **parent_variables):
        self.setup_embedding_parent(model_dim, parent_variables)
        self.transformer = PairEncoder(num_layers, num_heads, model_dim,
                                       FF_hidden_dim, self.device).to(self.device)
        self.finish_setup()

    def embedding_parameters(self):
        return self.transformer.parameters()

    def get_graph_statics(self, mdp, train):
        with torch.set_grad_enabled(train):
            self.graph_embedding, self.node_embedding = self.transformer(mdp)
        return self.graph_embedding

    def checkpoint(self):
        result = super().checkpoint()
        result["transformer"] = self.transformer.state_dict()
        return result

    def load_checkpoint(self, checkpoint):
        super().load_checkpoint(checkpoint)
        self.transformer.load_state_dict(checkpoint["transformer"])


class GNNTransformerA2C(EmbeddingA2C):
    def __init__(self, input_dim_gnn, gnn_hidden_dim, gnn_output_dim,
                 num_layers, large_value, num_heads, model_dim, FF_hidden_dim,
                 **parent_variables):
        embedding_dim = gnn_output_dim - 3
        if model_dim != embedding_dim:
            raise ValueError("model_dim must equal the GNN embedding dimension")
        self.setup_embedding_parent(embedding_dim, parent_variables)
        self.gnn = SimpleGNN(input_dim_gnn, gnn_hidden_dim, gnn_output_dim,
                             self.device, large_value).to(self.device)
        self.transformer = GraphEncoder(num_layers, num_heads, model_dim,
                                        FF_hidden_dim).to(self.device)
        self.finish_setup()

    def embedding_parameters(self):
        return chain(self.gnn.parameters(), self.transformer.parameters())

    def get_graph_statics(self, mdp, train):
        with torch.set_grad_enabled(train):
            self.node_embedding = self.gnn(mdp)
            self.graph_embedding = self.transformer(self.node_embedding)
        return self.graph_embedding

    def checkpoint(self):
        result = super().checkpoint()
        result.update({"gnn": self.gnn.state_dict(),
                       "transformer": self.transformer.state_dict()})
        return result

    def load_checkpoint(self, checkpoint):
        super().load_checkpoint(checkpoint)
        self.gnn.load_state_dict(checkpoint["gnn"])
        self.transformer.load_state_dict(checkpoint["transformer"])
