import torch.nn as nn
import torch
from copy import deepcopy

from DQN_parent import DQN

class OnlyDQN(DQN):
    def __init__(self, **parent_variebles):
        super().__init__(**parent_variebles)

        self.optimizer = torch.optim.Adam( # abetter optimizer what about the parameters 
            self.explore_model.parameters(), 
            lr = self.lr
        )

    def get_graph_statics(self, mdp, train):
        '''
        graph statics:
            - graph number of nodes -> 1
            - graph average distance -> 1
            - graph std distances -> 1
            - car capcities -> 1
            - nodes demands average -> 1
            - nodes std demands -> 1
        '''

        num_nod = torch.tensor([mdp.num_nodes], device=self.device, dtype=torch.float32)
        ave_dis = torch.tensor([sum(mdp.distance_matrix_ave) / len(mdp.distance_matrix_ave)],
                            device=self.device, dtype=torch.float32)
        std_dis = torch.tensor([mdp.distance_matrix_std], device=self.device, dtype=torch.float32)
        car_cap = torch.tensor([mdp.cars_capacity], device=self.device, dtype=torch.float32)
        dem_ave = torch.tensor([mdp.node_dem_ave], device=self.device, dtype=torch.float32)
        dem_std = torch.tensor([mdp.node_dem_std], device=self.device, dtype=torch.float32)

        return torch.cat([num_nod, ave_dis, std_dis, car_cap, dem_ave, dem_std])

    def get_unused_nodes_statics(self, mdp, unused):
        unused_list = list(unused)
        frac_unused = torch.tensor([len(unused) / mdp.num_nodes], device=self.device, dtype=torch.float32)
        ave_unused_cap = torch.tensor([sum(mdp.node_demand[unused_list]) / len(unused)], 
                                      device=self.device, dtype=torch.float32)

        return torch.cat([frac_unused, ave_unused_cap])

    def get_next_node_statics(self, mdp, route, next_node):
        '''                
        next node:
            - nodes capacity -> 1
            - current link to it distnace from current route -> 1
        '''
        sum_dis = torch.tensor([sum(mdp.distance_matrix[i][next_node] 
                                   for i in range(mdp.num_nodes) if i != next_node)],
                                   device=self.device, dtype=torch.float32)
        dis = torch.tensor([mdp.distance_matrix[route["current_node"]][next_node]], device=self.device, dtype=torch.float32)
        cap = torch.tensor([mdp.node_demand[next_node]], device=self.device, dtype=torch.float32)

        return torch.cat([sum_dis, cap, dis])




            
