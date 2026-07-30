import torch.nn as nn

class SimpleGNN(nn.Model):# if inherent from messagepassing would it be better?
    
    def __init__(self, node_dim, hidden_dim, output_dim, edge_dim):
        super().__init__(aggr='sum')

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
        message = sum(messages)
        node_neihbors = cat(message+nodes)
        return (self.message_aggregating(node_neihbors))