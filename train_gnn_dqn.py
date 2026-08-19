import matplotlib.pyplot as plt
import torch
import yaml

from GNNDQN.gnn_DQN import GNNDQN
from gnntransDQN.gnntransDQN import GNNTransDQN
from heuristic import ClarkeWrightSavings
from MDP import MDP


def total_dis(routes, mdp):
    overall_total_dis = 0
    for route in routes:
        route = route[0]
        route_distance = mdp.distance_matrix[mdp.depot_num][route[0]]
        for i in range(len(route) - 1):
            route_distance += mdp.distance_matrix[route[i]][route[i + 1]]
        route_distance += mdp.distance_matrix[route[-1]][mdp.depot_num]
        overall_total_dis += route_distance
    return overall_total_dis

def train_gnn_dqn():

    with open("conf.yaml", "r") as file:
        config = yaml.safe_load(file)

    # common hyperparameters (used to construct models)
    output_dim = config["output_dim"]
    hidden_dim = config["hidden_dim"]
    lr = config["lr"]
    lr_decay = config["lr_decay"]
    min_lr = config["min_lr"]
    target_update_counter = config["target_update_counter"]
    explore_update_counter = config["explore_update_counter"]
    discount = config["discount"]
    epsilon = config["epsilon"]
    epsilon_decay = config["epsilon_decay"]
    min_epsilon = config["min_epsilon"]
    batch_size = config["batch_size"]
    RB_capacity = config["RB_capacity"]
    num_first_samples = config["num_first_samples"]
    num_epoches = config["num_epoches"]
    max_num_nodes = config["max_num_nodes"]
    min_num_nodes = config["min_num_nodes"]
    depot_num = config["depot_num"]
    max_num_cars = config["max_num_cars"]
    min_num_cars = config["min_num_cars"]
    cars_capacity = config["cars_capacity"]
    min_distance = config["min_distance"]
    max_distance = config["max_distance"]
    min_node_dem = config["min_node_dem"]
    max_node_dem = config["max_node_dem"]
    max_grad_norm = config["max_grad_norm"]

    common = {
        "output_dim": output_dim,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "lr_decay": lr_decay,
        "min_lr": min_lr,
        "target_update_counter": target_update_counter,
        "explore_update_counter": explore_update_counter,
        "discount": discount,
        "epsilon": epsilon,
        "epsilon_decay": epsilon_decay,
        "min_epsilon": min_epsilon,
        "batch_size": batch_size,
        "RB_capacity": RB_capacity,
        "num_first_samples": num_first_samples,
        "num_epoches": num_epoches,
        "min_num_nodes": min_num_nodes,
        "max_num_nodes": max_num_nodes,
        "depot_num": depot_num,
        "min_num_cars": min_num_cars,
        "max_num_cars": max_num_cars,
        "cars_capacity": cars_capacity,
        "min_distance": min_distance,
        "max_distance": max_distance,
        "min_node_dem": min_node_dem,
        "max_node_dem": max_node_dem,
        "max_grad_norm": max_grad_norm,
    }

    # instantiate GNNDQN only (train script pattern matches other train_* files)
    comp = config.get("comparison", {})
    gnn_conf = comp.get("gnn_dqn", {})
    gnn_dqn = GNNDQN(
        input_dim_gnn=gnn_conf.get("input_dim", 3),
        gnn_hidden_dim=gnn_conf.get("hidden_dim", 16) if gnn_conf.get("hidden_dim") is None else gnn_conf.get("hidden_dim", 16),
        gnn_output_dim=gnn_conf.get("embedding_dim", 8) + 3 if gnn_conf.get("embedding_dim") is not None else gnn_conf.get("gnn_output_dim", 11),
        large_value=gnn_conf.get("large_value", 100),
        input_dim=3 * (gnn_conf.get("embedding_dim", 8)) + 11,
        **common,
    )

    # Train and save
    gnn_dqn.DQN_train()
    torch.save(
        {
            "explore_model": gnn_dqn.explore_model.state_dict(),
            "state_net": gnn_dqn.state_net.state_dict(),
            "action_net": gnn_dqn.action_net.state_dict(),
            "gnn": gnn_dqn.gnn.state_dict() if hasattr(gnn_dqn, "gnn") else None,
        },
        config["comparison"]["checkpoints"]["gnn_dqn"],
    )

    # Evaluation for the trained GNNDQN (no local search)
    heuristic_trials = []
    gnn_trials = []
    for n_nodes in range(min_num_nodes, min(max_num_nodes + 1, min_num_nodes + 5)):
        for n_cars in range(min_num_cars, max_num_cars + 1):
            test_mdp = MDP(n_nodes, depot_num, n_cars, cars_capacity)
            test_mdp.fill_distance_matrix(min_distance, max_distance)
            test_mdp.fill_node_dem_matrix(min_node_dem, max_node_dem)

            cws = ClarkeWrightSavings(test_mdp)
            feasible = cws.CWS_solve()
            if not feasible:
                continue

            heuristic_trial_dis = total_dis(cws.list_routes, test_mdp)
            heuristic_trials.append(heuristic_trial_dis)
            print("Heuristic routes:", cws.list_routes)

            gnn_total_dis, gnn_routes = gnn_dqn.evaluate(test_mdp)
            if len(gnn_dqn.used) == test_mdp.num_nodes:
                gnn_trials.append(gnn_total_dis)
                print(gnn_routes)
            else:
                raise RuntimeError("GNNDQN failed to visit every customer")

    plt.plot(heuristic_trials, label="heuristic")
    plt.plot(gnn_trials, label="gnn dqn")

    plt.legend()
    plt.savefig("gnn_node.png")
