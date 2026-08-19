import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import statistics

class Layer(nn.Module):
    def __init__(self, num_heads, model_dim, FF_hidden_dim):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                "model_dim must be divisible by num_heads"
            )
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        

        self.Q_matrix = nn.Linear(model_dim, model_dim, bias=False)
        self.K_matrix = nn.Linear(model_dim, model_dim, bias=False)
        self.V_matrix = nn.Linear(model_dim, model_dim, bias=False)
        self.O_proj = nn.Linear(model_dim, model_dim, bias=False)

        self.FF = nn.Sequential(
            nn.Linear(model_dim, FF_hidden_dim),
            nn.ReLU(),
            nn.Linear(FF_hidden_dim, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def MHA(self, input):
        num_nodes, _ = input.size()

        Q = self.Q_matrix(input)
        K = self.K_matrix(input)
        V = self.V_matrix(input)

        Q = Q.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)

        K_T = K.transpose(1, 2)

        score = F.softmax((Q @ K_T) / math.sqrt(self.head_dim), dim=-1)
        score = score @ V

        result = score.transpose(-3, -2).reshape(num_nodes, self.model_dim)
        result = self.O_proj(result)

        return result

    def forward(self, input):
        attn_out = self.MHA(self.norm1(input))
        input = input + attn_out

        ff_out = self.FF(self.norm2(input))
        # add norm layer here
        input = input + ff_out

        return input


class Encoder(nn.Module):
    def __init__(self,  num_layers, num_heads, model_dim, FF_hidden_dim, device):
        super().__init__()

        self.device = device
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            Layer(num_heads, model_dim, FF_hidden_dim) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(model_dim)

    def two_nodes_info(self, distance, node1, node2):
        distance_scale = max(
            float(sum(self.mdp.distance_matrix_ave))
            / max(self.mdp.num_nodes, 1),
            1e-6,
        )
        capacity_scale = max(float(self.mdp.cars_capacity), 1e-6)
        return torch.tensor(
            [
                self.mdp.distance_matrix_ave[node1] / distance_scale,
                self.mdp.node_demand[node1] / capacity_scale,
                self.mdp.distance_matrix_ave[node2] / distance_scale,
                self.mdp.node_demand[node2] / capacity_scale,
                distance / distance_scale,
            ],
            device=self.device,
            dtype=torch.float32,
        )

    def input_create(self):
        if self.mdp.num_nodes < 2:
            raise ValueError("Transformer encoder needs at least two nodes")

        pair_info = []
        self.pair_nodes = []
        for i in range(self.mdp.num_nodes):
            for j in range(i + 1, self.mdp.num_nodes):
                pair_info.append(
                    self.two_nodes_info(
                        self.mdp.distance_matrix[i][j], i, j
                    )
                )
                self.pair_nodes.append((i, j))

        return torch.stack(pair_info)

    def forward(self, mdp):
        self.mdp = mdp
        input = self.input_create()

        for layer in self.layers:
            input = layer(input)

        pair_embedding = self.final_norm(input)

        node_indices = torch.tensor(
            self.pair_nodes,
            device=self.device,
            dtype=torch.long,
        ).reshape(-1)
        repeated_pairs = pair_embedding.repeat_interleave(2, dim=0)
        node_embedding = torch.zeros(
            self.mdp.num_nodes,
            pair_embedding.size(-1),
            device=self.device,
            dtype=pair_embedding.dtype,
        ).index_add(0, node_indices, repeated_pairs)
        counts = torch.zeros(
            self.mdp.num_nodes,
            device=self.device,
            dtype=pair_embedding.dtype,
        ).index_add(
            0,
            node_indices,
            torch.ones_like(node_indices, dtype=pair_embedding.dtype),
        )
        node_embedding = node_embedding / counts.clamp_min(1.0).unsqueeze(-1)
        graph_embedding = node_embedding.mean(dim=0)

        return graph_embedding, node_embedding
