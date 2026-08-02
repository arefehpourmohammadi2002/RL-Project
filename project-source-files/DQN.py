import torch.nn as nn

class DQNetwork(nn.Module):
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


class DQN:
    def __init__(self, mdp, graph_embedding, node_embedding):
        self.routes = {
        f"route{k}": {
            "path": [mdp.depot_num],          
            "capacity": mdp.cars_capacity,      
            "total_distance": 0.0, 
            "current_node": mdp.depot_num 
        }
        for k in range(0, mdp.num_cars)
    }



