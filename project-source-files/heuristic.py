
class ClarkeWrightSavings:
    def __init__(self, MDP):
        self.MDP = MDP
        self.depot = MDP.depot_num
        self.num_cars = MDP.num_cars
        ''' here needs to be adjusted'''
        self.list_routs = list(range(1, MDP.num_nodes)) # this is the fist clustring of CWS
        self.saving = []

    def saving_two_routs(self, route1, route2):
        route2_first_node = route2[0]
        route1_last_node = route1[-1]

        route_1_profit = self.MDP[self.depot][route2_first_node]
        route_2_profit = self.MDP[route2_first_node][self.depot]
        cost_of_this_merge = self.MDP[route1_last_node][route2_first_node]

        return route_1_profit + route_2_profit - cost_of_this_merge

    def sort(self):
        self.saving.sort(reverse=True)

    def merge(self):
        used = []
        routs_num = 0
        temp_routes_list = []
        stabel = True

        for i in range(len(self.saving)):
            current_saving = self.saving[i][0]
            first_rout = self.saving[i][1]
            second_rout = self.saving[i][2]

            if first_rout not in used and second_rout not in used:
                if current_saving >= 0 or (len(self.list_routs) - i + routs_num > self.num_cars):
                    new_route = self.list_routs[i[1]] + self.list_routs[i[2]]
                    temp_routes_list.append(new_route)
                    routs_num += 1 
                    used.append(i[0])
                    used.append(i[1])
                    stabel = False

                else:
                    temp_routes_list.append(self.list_routs[i[1]])
                    temp_routes_list.append(self.list_routs[i[2]])
                    routs_num += 2 
                    used.append(i[0])
                    used.append(i[1])
                    
        self.list_routs = temp_routes_list
        if routs_num > self.num_cars:
            return False
        return stabel
            

    def saving_calculate(self):
        for i in len(self.list_routs):
            for j in len(self.list_routs):
                saving = self.saving_two_routs(self.list_routs[i], self.list_routs[j])
                self.saving.append([saving, i , j])
    
    def CWS_solve(self):
        while True:
            self.saving_calculate()
            self.sort()
            stabel = self.merge()
            if stabel: 
                break
        
