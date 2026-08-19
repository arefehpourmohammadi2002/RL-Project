
import math
import pickle
import random

import numpy as np

from DQN_parent import DQN
from MDP import MDP
from heuristic import ClarkeWrightSavings


def heuristic_distance(routes, mdp):
    total = 0.0
    for route, _load in routes:
        path = [mdp.depot_num, *route, mdp.depot_num]
        total += sum(
            mdp.distance_matrix[first][second]
            for first, second in zip(path, path[1:])
        )
    return float(total)


def build_instance(rng, num_nodes, num_cars, comparison):
    while True:
        np.random.seed(rng.randrange(2**32))
        mdp = MDP(
            num_nodes,
            comparison["depot_num"],
            num_cars,
            comparison["cars_capacity"],
        )
        mdp.build(
            comparison["min_distance"],
            comparison["max_distance"],
            comparison["min_node_demand"],
            comparison["max_node_demand"],
        )
        demands = [
            mdp.node_demand[node]
            for node in range(mdp.num_nodes)
            if node != mdp.depot_num
        ]
        if not DQN.can_pack_demands(
            demands, [mdp.cars_capacity] * mdp.num_cars
        ):
            continue
        heuristic = ClarkeWrightSavings(mdp)
        if heuristic.CWS_solve():
            return mdp, heuristic_distance(heuristic.list_routes, mdp)


def cars_for_node_count(num_nodes, comparison):

    node_sweep = comparison["node_sweep"]
    avg_demand = (comparison["min_node_demand"] + comparison["max_node_demand"]) / 2
    expected_total_demand = (num_nodes - 1) * avg_demand
    required_cars = math.ceil(expected_total_demand / comparison["cars_capacity"]) + 1
    return max(node_sweep["min_cars"], min(required_cars, node_sweep["max_cars"]))


def generate_test_samples(comparison):

    node_values = list(
        range(
            comparison["node_sweep"]["min_nodes"],
            comparison["node_sweep"]["max_nodes"] + 1,
            comparison["node_sweep"]["step"],
        )
    )
    car_values = list(
        range(
            comparison["car_sweep"]["min_cars"],
            comparison["car_sweep"]["max_cars"] + 1,
            comparison["car_sweep"]["step"],
        )
    )
    trials = comparison["trials_per_point"]
    rng = random.Random(comparison["seed"])

    node_instances = {
        value: [
            build_instance(
                rng, value, cars_for_node_count(value, comparison), comparison
            )
            for _trial in range(trials)
        ]
        for value in node_values
    }
    car_instances = {
        value: [
            build_instance(
                rng, comparison["car_sweep"]["fixed_nodes"], value, comparison
            )
            for _trial in range(trials)
        ]
        for value in car_values
    }
    return {"nodes": node_instances, "cars": car_instances}


def save_test_samples(test_samples, path):
    with open(path, "wb") as sample_file:
        pickle.dump(test_samples, sample_file)


def load_test_samples(path):
    with open(path, "rb") as sample_file:
        return pickle.load(sample_file)
