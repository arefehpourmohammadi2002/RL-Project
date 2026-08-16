import torch.nn as nn
import torch
from itertools import chain

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, large_value):
        super().__init__()

        if input_dim != 3:
            raise ValueError(
                "SimpleGNN creates three edge-message features, "
                f"so input_dim must be 3, got {input_dim}"
            )
        if output_dim <= 3:
            raise ValueError("output_dim must be greater than 3")

        self.device = device
        self.large_value = large_value

        self.message_passing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.message_aggregating = nn.Sequential(
            nn.Linear(output_dim, output_dim-3)
        )

    def forward(self, mdp):

        self.mdp = mdp
        node_fet = self.create_node_feature()
        edge_fet = self.create_edge_feature()
        input = self.create_input(node_fet, edge_fet)

        messages = self.message_passing(input)
        messages = messages.mean(dim=1)

        return self.message_aggregating(messages)


    def create_node_feature(self):
        return self.mdp.node_demand


    def create_edge_feature(self):
        return self.mdp.distance_matrix


    def create_input(self, node_feature, edge_feature):
        node_feature = torch.as_tensor(node_feature, dtype=torch.float32, device=self.device)
        edge_feature = torch.as_tensor(edge_feature, dtype=torch.float32, device=self.device)

        node_mean = node_feature.mean()
        node_std = node_feature.std(unbiased=False).clamp_min(1e-6)
        node_feature = (node_feature - node_mean) / node_std

        finite_mask = torch.isfinite(edge_feature)
        if not finite_mask.any():
            raise ValueError("distance matrix has no finite values")
        edge_mean = edge_feature[finite_mask].mean()
        edge_std = edge_feature[finite_mask].std(
            unbiased=False
        ).clamp_min(1e-6)
        edge_feature = (edge_feature - edge_mean) / edge_std

        num_nodes = node_feature.size(0)
        source_feature = node_feature[:, None].expand(num_nodes, num_nodes)
        target_feature = node_feature[None, :].expand(num_nodes, num_nodes)
        input = torch.stack(
            [source_feature, target_feature, edge_feature], dim=-1
        )
        return torch.nan_to_num(
            input,
            nan=0.0,
            posinf=self.large_value,
            neginf=-self.large_value,
        )
