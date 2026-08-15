import torch.nn as nn
import torch
from itertools import chain

class SimpleGNN(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, edge_dim, device, large_value):
        super().__init__()

        self.device = device
        self.large_value = large_value

        input_dim = node_dim + node_dim + edge_dim

        self.message_passing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.message_aggregating = nn.Sequential(
            nn.Linear(output_dim + node_dim, output_dim)
        )

    def forward(self, mdp):

        self.mdp = mdp
        node_fet = self.create_node_feature()
        edge_fet = self.create_edge_feature()
        input = self.create_input(node_fet, edge_fet)

        messages = self.message_passing(input)
        messages = messages.mean(dim=1)

        node_feat = input[:, 0, 0:1]
        combine = torch.cat([messages, node_feat], dim=-1)

        return self.message_aggregating(combine)


    def create_node_feature(self):
        return self.mdp.node_demand


    def create_edge_feature(self):
        return self.mdp.distance_matrix


    def create_input(self, node_feature, edge_feature):
        node_feature = torch.as_tensor(node_feature, dtype=torch.float32, device=self.device)
        edge_feature = torch.as_tensor(edge_feature, dtype=torch.float32, device=self.device)

        node_mean = node_feature.mean()
        node_std = node_feature.std() + 1e-6
        node_feature = (node_feature - node_mean) / node_std

        finite_mask = torch.isfinite(edge_feature)
        edge_mean = edge_feature[finite_mask].mean()
        edge_std = edge_feature[finite_mask].std() + 1e-6
        edge_feature = (edge_feature - edge_mean) / edge_std

        num_nodes = node_feature.size(0)
        input = torch.zeros(num_nodes, num_nodes, 3, device=self.device)
        for i in range(num_nodes):
            for j in range(num_nodes):
                input[i, j, 0] = node_feature[i]
                input[i, j, 1] = node_feature[j]
                input[i, j, 2] = edge_feature[i, j]
        input = torch.where(torch.isinf(input), torch.full_like(input, self.large_value), input)
        return input