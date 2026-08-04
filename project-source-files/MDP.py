import copy
import numpy as np


class MDP():
    def __init__(self, num_nodes, depot_num, num_cars, cars_capacity, distance_matrix=None, node_capacity=None):
        self.num_nodes = num_nodes
        self.depot_num = depot_num
        self.num_cars = num_cars
        self.cars_capacity = cars_capacity
        self.distance_matrix = distance_matrix
        self.node_capacity = node_capacity

    def fill_distance_matrix(self, min_distance, max_distance):

        self.distance_matrix = np.random.uniform(
            min_distance,
            max_distance,
            size=(self.num_nodes, self.num_nodes)
        )
        for i in range(self.num_nodes):
            self.distance_matrix[i][i] = float("inf")

    def fill_node_cap_matrix(self, min_node_cap, max_node_cap):

        self.node_capacity = np.random.uniform(
            min_node_cap,
            max_node_cap,
            size=(self.num_nodes)
        )


def apply_action(route, action, mdp):

    new_route = copy.deepcopy(route)
    if action is None:
        return new_route
    if new_route["capacity"] + mdp.node_capacity[action] <= mdp.cars_capacity:
        new_route["path"].append(action)
        new_route["capacity"] += mdp.node_capacity[action]
        new_route["total_distance"] += mdp.distance_matrix[new_route["current_node"], action]
        new_route["current_node"] = action
    return new_route