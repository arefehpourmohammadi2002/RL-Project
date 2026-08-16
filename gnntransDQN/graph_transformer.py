import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self,  num_layers, num_heads, model_dim, FF_hidden_dim):
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            Layer(num_heads, model_dim, FF_hidden_dim) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, input):

        for layer in self.layers:
            input = layer(input)

        input = self.final_norm(input)
        graph_embedding = input.mean(dim=0)

        return graph_embedding