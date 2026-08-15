import numpy as np
import statistics


class MDP:
    def __init__(self, num_nodes, depot_num, num_cars, cars_capacity):
        self.num_nodes = num_nodes
        self.depot_num = depot_num
        self.num_cars = num_cars
        self.cars_capacity = cars_capacity

        self.distance_matrix = None
        self.distance_matrix_ave = None
        self.distance_matrix_std = None

        self.node_demand = None
        self.node_dem_ave = None
        self.node_dem_std = None

    def fill_distance_matrix(self, min_distance, max_distance):
        self.distance_matrix = np.random.uniform(
            min_distance,
            max_distance,
            size=(self.num_nodes, self.num_nodes)
        )
        self.distance_matrix = (self.distance_matrix + self.distance_matrix.T) / 2
        for i in range(self.num_nodes):
            self.distance_matrix[i][i] = 0.0

        node_dis_sum = [sum(self.distance_matrix[:, i]) for i in range(self.num_nodes)]
        self.distance_matrix_ave = [node_dis_sum[i] / (self.num_nodes - 1) for i in range(self.num_nodes)]
        self.distance_matrix_std = statistics.pstdev(
            node_dis_sum[i] / (self.num_nodes - 1)
            for i in range(self.num_nodes)
            if i != self.depot_num
        )

    def fill_node_dem_matrix(self, min_node_dem, max_node_dem):
        self.node_demand = np.random.uniform(
            min_node_dem,
            max_node_dem,
            size=(self.num_nodes)
        )
        self.node_demand[self.depot_num] = 0

        node_dem_sum = sum(self.node_demand[i] for i in range(self.num_nodes) if i != self.depot_num)
        self.node_dem_ave = node_dem_sum / (self.num_nodes - 1)
        self.node_dem_std = statistics.pstdev(
            self.node_demand[i]
            for i in range(self.num_nodes)
            if i != self.depot_num
        )

    def build(self, min_distance, max_distance, min_node_dem, max_node_dem):
        self.fill_distance_matrix(min_distance=min_distance, max_distance=max_distance)
        self.fill_node_dem_matrix(min_node_dem=min_node_dem, max_node_dem=max_node_dem)
        return self