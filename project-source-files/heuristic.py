
class ClarkeWrightSavings:
    def __init__(self, MDP):
        self.MDP = MDP
        self.depot = MDP.depot_num
        self.num_cars = MDP.num_cars
        self.car_cap = MDP.cars_capacity
        self.num_routes = MDP.num_nodes

        self.list_routes = [] # this is the fist clustring of CWS
        for i in range(self.MDP.num_nodes):
            if i != self.depot:
                j = self.MDP.node_capacity[i]
                temp_list = [[i], j]
                self.list_routes.append(temp_list)

        self.savings = []

    def saving_two_routes(self, route1_id, route2_id):

        route1 = self.list_routes[route1_id][0]
        route2 = self.list_routes[route2_id][0]

        route2_first_node = route2[0]
        route1_last_node = route1[-1]

        routee_1_profit = self.MDP.distance_matrix[route1_last_node][self.depot]
        routee_2_profit = self.MDP.distance_matrix[self.depot][route2_first_node]
        cost_of_this_merge = self.MDP.distance_matrix[route1_last_node][route2_first_node]

        return routee_1_profit + routee_2_profit - cost_of_this_merge

    def sort(self):
        self.savings.sort(reverse=True)

    def merge(self):
        used = []
        num_routes = 0
        stabel = True

        for i in range(len(self.savings)):

            current_saving = self.savings[i][0]
            first_route_id = self.savings[i][1]
            second_route_id = self.savings[i][2]

            route1_cap = self.list_routes[first_route_id][1]
            route2_cap = self.list_routes[second_route_id][1]

            if route1_cap + route2_cap <= self.car_cap:

                if first_route_id not in used and second_route_id not in used:
                    if (current_saving >= 0 
                        or (len(self.list_routes) - i + num_routes > self.num_cars) and stabel):

                        new_route = self.list_routes[first_route_id][0] + self.list_routes[second_route_id][0]
                        new_cap = route1_cap + route2_cap
                        new_route = [new_route, new_cap]

                        self.list_routes.append(new_route)

                        used.append(first_route_id)
                        used.append(second_route_id)

                        num_routes += 1 
                        stabel = False

                    else:
                        break

                else:
                    break
                
        for route_id in sorted(used, reverse=True):
            del self.list_routes[route_id]

        return stabel
            
    def saving_calculate(self):

        for i in range(len(self.list_routes)):
            for j in range(len(self.list_routes)):
                if i != j:
                    saving = self.saving_two_routes(i, j)
                    self.savings.append([saving, i , j])

    
    def CWS_solve(self):

        while True:
            self.savings.clear()
            self.saving_calculate()
            self.sort()
            stabel = self.merge()

            if stabel:
                if len(self.list_routes) > self.num_cars:
                    return False
                return True

                
        
