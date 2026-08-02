import torch.nn as nn

class DQNCVRP(nn.Module):
    def __init__(self, graph_embedding, current_node, next_node, current_car_cap, hidden_dim, output_dim):
        super().__init__()

        input_size = len(graph_embedding) + len(current_node) + len(next_node) + len(current_car_cap)

        self.dqn = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input):
        return self.dqn(input)

    