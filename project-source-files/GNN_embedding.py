import torch.nn as nn
import torch

class SimpleGNN(nn.Module):# if inherent from messagepassing would it be better?
    
    def __init__(self, node_dim, hidden_dim, output_dim, edge_dim):
        super().__init__()

        input = node_dim + node_dim + edge_dim

        self.message_passing = nn.Sequential(

            nn.Linear(input, hidden_dim),
            nn.ReLU(), # is ReLU proper ?
            nn.Linear(hidden_dim, output_dim)
            # does as activation layer needed here?
        )

        self.message_aggregating = nn.Sequential(
    
            nn.Linear(output_dim+node_dim, output_dim),
            nn.ReLU()
            # does the relu is ok more layers is needed?
        )

    def forward(self, input):
        messages = self.message_passing(input)
        messages = messages.sum(dim=1)

        node_feat = input[:, 0, 0:1]             
        combine = torch.cat([messages, node_feat], dim=-1)

        return (self.message_aggregating(combine))


def create_node_feature(mdp):
    return mdp.node_capacity

def create_edge_feature(mdp):
    return mdp.distance_matrix

def create_input(node_feature, edge_feature):
    
    input = torch.zeros(len(node_feature), len(node_feature), 3)
    for i in range(len(node_feature)):
        for j in range(len(node_feature)):
            input[i, j, 0] = node_feature[i]
            input[i, j, 1] = node_feature[j]
            input[i, j, 2] = edge_feature[i, j]

    return input 
