import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import statistics

class Layer(nn.Module):
    def __init__(self, num_heads, model_dim, FF_hidden_dim):
        super().__init__()
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

    ''' model dim is 5, determined by this function output dimention'''
    def two_nodes_info(self, distance, node1, node2):
        dis = self.mdp.distance_matrix[node1][node2]

        node1_ave_dis = self.mdp.distance_matrix_ave[node1]
        node2_ave_dis = self.mdp.distance_matrix_ave[node2]

        node1_ave_std = self.mdp.distance_matrix_std[node1]
        node2_ave_std = self.mdp.distance_matrix_std[node2]

        return torch.cat([node1_ave_dis, node1_ave_std, node2_ave_std, node2_ave_dis, distance]
                  , device=self.device, dtype=torch.float)

    def input_create(self):
        pair_info = []
        for i in range(self.mdp.num_nodes):
            for j in range(i+1, self.mdp.num_nodes):
                pair_info.append(self.two_nodes_info(self.mdp.distance_matrix[i][j], i, j))

        return torch.stack(pair_info, device=self.device, dtype=torch.float)

    def forward(self, mdp):
        self.mdp = mdp
        input = self.input_create()

        for layer in self.layers:
            input = layer(input)

        node_embedding = self.final_norm(input)
        graph_embedding = node_embedding.mean(dim=0)

        return graph_embedding, node_embedding