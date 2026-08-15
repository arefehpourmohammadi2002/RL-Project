import torch.nn as nn
import torch
from copy import deepcopy

from ..DQN_parent import DQN

class OnlyDQN(DQN):
    def __init__(self, lr, **parent_variebles):
        super().__init__(**parent_variebles)

        self.optimizer = torch.optim.Adam( # abetter optimizer what about the parameters 
            self.explore_model.parameters(), 
            lr = lr
        )

    def get_graph_statics(self, mdp):
        '''
        graph statics:
            - graph number of nodes -> 1
            - graph average distance -> 1
            - graph std distances -> 1
            - car capcities -> 1
            - nodes demands average -> 1
            - nodes std demands -> 1
        '''

        num_nod = torch.tensor(mdp.num_nod, device=self.device, dtype=float)
        ave_dis = torch.tensor(mdp.ave_dis, device=self.device, dtype=float)
        std_dis = torch.tensor(mdp.std_dis, device=self.device, dtype=float)
        car_cap = torch.tensor(mdp.car_cap, device=self.device, dtype=float)
        dem_ave = torch.tensor(mdp.dem_ave, device=self.device, dtype=float)
        std_nod = torch.tensor(mdp.std_nod, device=self.device, dtype=float)

        torch.cat([num_nod, ave_dis, std_dis, car_cap, dem_ave, std_nod])





            
