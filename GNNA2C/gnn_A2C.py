import torch
from A2C_parent import A2C
from GNNDQN.GNN_embedding import SimpleGNN


class GNNA2C(A2C):
    def __init__(self, input_dim_gnn, gnn_hidden_dim, gnn_output_dim, large_value, **parent_variables):
        parent_variables.setdefault("actor_input_dim", 3 * (gnn_output_dim - 3) + 5)
        parent_variables.setdefault("critic_input_dim", 2 * (gnn_output_dim - 3) + 3)
        super().__init__(**parent_variables)
        self.gnn = SimpleGNN(input_dim=input_dim_gnn, hidden_dim=gnn_hidden_dim, output_dim=gnn_output_dim, device=self.device, large_value=large_value)
        self.gnn.to(self.device)
        self.node_embedding = None

    def get_graph_statics(self, mdp, train):
        if not train:
            with torch.no_grad():
                self.node_embedding = self.gnn(mdp)
        else:
            self.node_embedding = self.gnn(mdp)
        return self.node_embedding.mean(dim=0)

    def get_unused_nodes_statics(self, mdp, used):
        if self.node_embedding is None:
            return torch.zeros_like(self.node_embedding[0])
        unused_embeddings = [self.node_embedding[i] for i in range(mdp.num_nodes) if i not in used]
        if not unused_embeddings:
            return torch.zeros_like(self.node_embedding[0])
        return torch.stack(unused_embeddings).mean(dim=0)

    def get_next_node_statics(self, mdp, route, next_node):
        if self.node_embedding is None:
            raise RuntimeError("node embeddings must be created first")
        distance = mdp.distance_matrix[route["current_node"]][next_node] / self.distance_scale(mdp)
        distance = torch.tensor([distance], device=self.device, dtype=torch.float32)
        return torch.cat([self.node_embedding[next_node], distance])
