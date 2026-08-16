def route_distance(customers, mdp):
    path = [mdp.depot_num, *customers, mdp.depot_num]
    return sum(
        mdp.distance_matrix[first][second]
        for first, second in zip(path, path[1:])
    )


def improve_routes(routes, mdp):
    """Capacity-safe 2-opt, relocate, and swap improvement."""
    customer_routes = [
        [node for node in route["path"] if node != mdp.depot_num]
        for route in routes
    ]
    loads = [
        sum(mdp.node_demand[node] for node in route)
        for route in customer_routes
    ]
    tolerance = 1e-10

    improved = True
    while improved:
        improved = False

        for route in customer_routes:
            old_distance = route_distance(route, mdp)
            best_delta = 0.0
            best_pair = None
            for first in range(len(route) - 1):
                for second in range(first + 1, len(route)):
                    candidate = (
                        route[:first]
                        + list(reversed(route[first:second + 1]))
                        + route[second + 1:]
                    )
                    delta = route_distance(candidate, mdp) - old_distance
                    if delta < best_delta - tolerance:
                        best_delta = delta
                        best_pair = (first, second)
            if best_pair is not None:
                first, second = best_pair
                route[first:second + 1] = reversed(
                    route[first:second + 1]
                )
                improved = True

        best_delta = 0.0
        best_move = None
        for source_index, source in enumerate(customer_routes):
            for node_position, node in enumerate(source):
                demand = mdp.node_demand[node]
                for target_index, target in enumerate(customer_routes):
                    if (
                        source_index == target_index
                        or loads[target_index] + demand
                        > mdp.cars_capacity + tolerance
                    ):
                        continue
                    old_distance = (
                        route_distance(source, mdp)
                        + route_distance(target, mdp)
                    )
                    shorter_source = (
                        source[:node_position] + source[node_position + 1:]
                    )
                    for insertion_position in range(len(target) + 1):
                        longer_target = (
                            target[:insertion_position]
                            + [node]
                            + target[insertion_position:]
                        )
                        delta = (
                            route_distance(shorter_source, mdp)
                            + route_distance(longer_target, mdp)
                            - old_distance
                        )
                        if delta < best_delta - tolerance:
                            best_delta = delta
                            best_move = (
                                source_index,
                                node_position,
                                target_index,
                                insertion_position,
                                demand,
                            )

        if best_move is not None:
            (
                source_index,
                node_position,
                target_index,
                insertion_position,
                demand,
            ) = best_move
            node = customer_routes[source_index].pop(node_position)
            customer_routes[target_index].insert(insertion_position, node)
            loads[source_index] -= demand
            loads[target_index] += demand
            improved = True

        best_delta = 0.0
        best_swap = None
        for first_index in range(len(customer_routes)):
            for second_index in range(first_index + 1, len(customer_routes)):
                first_route = customer_routes[first_index]
                second_route = customer_routes[second_index]
                old_distance = (
                    route_distance(first_route, mdp)
                    + route_distance(second_route, mdp)
                )
                for first_position, first_node in enumerate(first_route):
                    for second_position, second_node in enumerate(second_route):
                        first_load = (
                            loads[first_index]
                            - mdp.node_demand[first_node]
                            + mdp.node_demand[second_node]
                        )
                        second_load = (
                            loads[second_index]
                            - mdp.node_demand[second_node]
                            + mdp.node_demand[first_node]
                        )
                        if (
                            first_load > mdp.cars_capacity + tolerance
                            or second_load > mdp.cars_capacity + tolerance
                        ):
                            continue
                        new_first = first_route.copy()
                        new_second = second_route.copy()
                        (
                            new_first[first_position],
                            new_second[second_position],
                        ) = (
                            second_node,
                            first_node,
                        )
                        delta = (
                            route_distance(new_first, mdp)
                            + route_distance(new_second, mdp)
                            - old_distance
                        )
                        if delta < best_delta - tolerance:
                            best_delta = delta
                            best_swap = (
                                first_index,
                                second_index,
                                first_position,
                                second_position,
                                first_load,
                                second_load,
                            )

        if best_swap is not None:
            (
                first_index,
                second_index,
                first_position,
                second_position,
                first_load,
                second_load,
            ) = best_swap
            (
                customer_routes[first_index][first_position],
                customer_routes[second_index][second_position],
            ) = (
                customer_routes[second_index][second_position],
                customer_routes[first_index][first_position],
            )
            loads[first_index] = first_load
            loads[second_index] = second_load
            improved = True

    improved_routes = []
    for customers, load in zip(customer_routes, loads):
        path = [mdp.depot_num, *customers]
        if customers:
            path.append(mdp.depot_num)
        improved_routes.append(
            {
                "path": path,
                "total_distance": float(route_distance(customers, mdp)),
                "capacity": float(load),
                "current_node": mdp.depot_num,
            }
        )
    return improved_routes
