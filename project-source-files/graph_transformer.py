import torch.nn as nn
import torch
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

    def MHA(self, input):
        num_nodes, _ = input.size()

        Q = self.Q_matrix(input)
        K = self.K_matrix(input)
        V = self.V_matrix(input)

        Q = Q.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)

        K_T = K.transpose(1, 2)

        score = F.softmax((Q @ K_T) / math.sqrt(self.head_dim)) 
        score = score @ V
        
        result = score.transpose(-3, -2).reshape(num_nodes, self.model_dim)
        result = self.O_proj(result)

        return result

    def norm():
        pass
    
    def forward(self, input):
        after_MHA = self.MHA(input)
        after_FF = self.FF(after_MHA)
        return self.norm(after_FF)
        

class Encoder(nn.Module):
    def __init__(self,  num_layers, simple_GNN_model, num_heads, model_dim, FF_hidden_dim):
        self.num_layers = num_layers
        self.simple_GNN_model = simple_GNN_model
        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(Layer( num_heads, model_dim, FF_hidden_dim))
    
    def forward(self, input):
        input = self.simple_GNN_model(input)
        for i in range(self.num_layers):
            input = self.layer[i](input)
        
        graph_embedding = sum(input)

        return graph_embedding


layer = Layer(3, 6)

input = torch.tensor([[1, 2, 3, 4, 5, 16],
                     [6, 7, 8, 9, 10, 17],
                     [11, 12, 13, 14, 15, 18],
                     [21, 22, 23, 24, 25, 28]], dtype=torch.float32)

layer.MHA(input=input)