import torch.nn as nn
import torch


class SimpleGNN(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, edge_dim):
        super().__init__()

        input_dim = node_dim + node_dim + edge_dim

        self.message_passing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.message_aggregating = nn.Sequential(
            nn.Linear(output_dim + node_dim, output_dim)
        )

    def forward(self, input):
        messages = self.message_passing(input)
        messages = messages.mean(dim=1)

        node_feat = input[:, 0, 0:1]
        combine = torch.cat([messages, node_feat], dim=-1)

        return self.message_aggregating(combine)


def create_node_feature(mdp):
    return mdp.node_demand


def create_edge_feature(mdp):
    return mdp.distance_matrix


def create_input(node_feature, edge_feature, large_value):
    node_feature = torch.as_tensor(node_feature, dtype=torch.float32)
    edge_feature = torch.as_tensor(edge_feature, dtype=torch.float32)

    node_mean = node_feature.mean()
    node_std = node_feature.std() + 1e-6
    node_feature = (node_feature - node_mean) / node_std

    finite_mask = torch.isfinite(edge_feature)
    edge_mean = edge_feature[finite_mask].mean()
    edge_std = edge_feature[finite_mask].std() + 1e-6
    edge_feature = (edge_feature - edge_mean) / edge_std

    num_nodes = node_feature.size(0)
    input = torch.zeros(num_nodes, num_nodes, 3)
    for i in range(num_nodes):
        for j in range(num_nodes):
            input[i, j, 0] = node_feature[i]
            input[i, j, 1] = node_feature[j]
            input[i, j, 2] = edge_feature[i, j]
    input = torch.where(torch.isinf(input), torch.full_like(input, large_value), input)
    return input