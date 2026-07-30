import torch.nn as nn
class Layer(nn.Model):
    def __init__(self, num_heads, input):
        super.__init__()
        self.num_heads = num_heads
        self.Q_matrix = []
        self.K_matrix = []
        self.V_matrix = []
        self.W_O_weight = []

        for _ in range(self.num_heads):
            self.Q_weight.append([])
            self.K_weight.append([])
            self.V_weight.append([])

        self.FF = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def MHSA(self, input):
        input_copy = input
        each_head = []
        for i in range(self.num_heads):
            Q = self.Q_weight[i] @ input_copy
            K = self.K_weight[i] @ input_copy
            V = self.V_weight[i] @ input_copy
            K_T = K.transpose()
            
            each_head.append(nn.softmax(Q @ K_T) @ V)
        
        return (cat(each_head) @ self.W_O_weight)
    
    def norm():
        pass
    
    def forward(self, input):
        after_MHSA = self.MHSA(input)
        after_FF = self.FF(after_MHSA)
        return self.norm(after_FF)
        

class Encoder(nn.Model):
    def __init__(self,  num_layers, simple_GNN_model):
        self.num_layers = num_layers
        self.simple_GNN_model = simple_GNN_model
        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(Layer(input))
    
    def forward(self, input):
        input = self.simple_GNN_model(input)
        for i in range(self.num_layers):
            input = self.layer[i](input)
        
        graph_embedding = sum(input)

        return graph_embedding


