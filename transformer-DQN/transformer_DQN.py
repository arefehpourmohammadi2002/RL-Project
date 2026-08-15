import torch.nn as nn
import torch
from copy import deepcopy

from transformer import Encoder
from ..DQN_parent import DQN
import replay_buffer as RB


class DQN:
    def __init__(self, lr, **parent_variebles):
        super().__init__(**parent_variebles)

        self.transforemer = Encoder()
        self.optimizer = torch.optim.Adam( # abetter optimizer what about the parameters 
            self.explore_model.parameters(), 
            self.transforemer.parameters(),
            lr=lr
        )

    def get_graph_statics(self, mdp, train):
        '''
        graph statics:
            is given from the transformer
        '''
        if not train:
            with torch.no_grad():
                return self.transforemer(mdp)
        return self.transforemer(mdp)





            
