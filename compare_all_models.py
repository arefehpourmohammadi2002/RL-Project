import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from DQN_parent import DQN
from GNNDQN.gnn_DQN import GNNDQN
from MDP import MDP
from gnntransDQN.gnntransDQN import GNNTransDQN
from heuristic import ClarkeWrightSavings
from onlyDQN.only_DQN import OnlyDQN
from transformerDQN.transformer_DQN import TransformerDQN


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


with open("conf.yaml", "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

comparison = config["comparison"]
random.seed(comparison["seed"])
np.random.seed(comparison["seed"])
torch.manual_seed(comparison["seed"])

common = {
    "output_dim": config["output_dim"],
    "hidden_dim": comparison["hidden_dim"],
    "lr": config["lr"],
    "lr_decay": config["lr_decay"],
    "min_lr": config["min_lr"],
    "target_update_counter": config["target_update_counter"],
    "explore_update_counter": config["explore_update_counter"],
    "discount": config["discount"],
    "epsilon": config["epsilon"],
    "epsilon_decay": config["epsilon_decay"],
    "min_epsilon": config["min_epsilon"],
    "batch_size": config["batch_size"],
    "RB_capacity": config["RB_capacity"],
    "num_first_samples": config["num_first_samples"],
    "num_epoches": config["num_epoches"],
    "min_num_nodes": comparison["training_min_nodes"],
    "max_num_nodes": comparison["training_max_nodes"],
    "depot_num": comparison["depot_num"],
    "min_num_cars": comparison["training_min_cars"],
    "max_num_cars": comparison["training_max_cars"],
    "cars_capacity": comparison["cars_capacity"],
    "min_distance": comparison["min_distance"],
    "max_distance": comparison["max_distance"],
    "min_node_dem": comparison["min_node_demand"],
    "max_node_dem": comparison["max_node_demand"],
    "max_grad_norm": config["max_grad_norm"],
}

gnn_config = comparison["gnn_dqn"]
transformer_config = comparison["transformer_dqn"]
hybrid_config = comparison["gnn_transformer_dqn"]

models = {
    "OnlyDQN": OnlyDQN(input_dim=OnlyDQN.INPUT_DIM, **common),
    "GNNDQN": GNNDQN(
        input_dim_gnn=gnn_config["input_dim"],
        gnn_hidden_dim=gnn_config["hidden_dim"],
        gnn_output_dim=gnn_config["embedding_dim"] + 3,
        large_value=gnn_config["large_value"],
        input_dim=3 * gnn_config["embedding_dim"] + 5,
        **common,
    ),
    "TransformerDQN": TransformerDQN(
        num_layers=transformer_config["num_layers"],
        num_heads=transformer_config["num_heads"],
        model_dim=transformer_config["model_dim"],
        FF_hidden_dim=transformer_config["feed_forward_dim"],
        input_dim=3 * transformer_config["model_dim"] + 5,
        **common,
    ),
    "GNN-TransformerDQN": GNNTransDQN(
        input_dim_gnn=hybrid_config["gnn_input_dim"],
        gnn_hidden_dim=hybrid_config["gnn_hidden_dim"],
        gnn_output_dim=hybrid_config["gnn_output_dim"] + 3,
        large_value=hybrid_config["large_value"],
        num_layers=hybrid_config["num_layers"],
        num_heads=hybrid_config["num_heads"],
        model_dim=hybrid_config["model_dim"],
        FF_hidden_dim=hybrid_config["feed_forward_dim"],
        input_dim=3 * hybrid_config["gnn_output_dim"] + 5,
        **common,
    ),
}

checkpoint_keys = {
    "OnlyDQN": "only_dqn",
    "GNNDQN": "gnn_dqn",
    "TransformerDQN": "transformer_dqn",
    "GNN-TransformerDQN": "gnn_transformer_dqn",
}
for name, model in models.items():
    checkpoint = torch.load(
        comparison["checkpoints"][checkpoint_keys[name]],
        map_location=model.device,
        weights_only=True,
    )
    explore_state = checkpoint.get("explore_model", checkpoint)
    model.explore_model.load_state_dict(explore_state)
    model.target_model.load_state_dict(explore_state)
    if hasattr(model, "gnn") and "gnn" in checkpoint:
        model.gnn.load_state_dict(checkpoint["gnn"])
    if hasattr(model, "transforemer") and "transformer" in checkpoint:
        model.transforemer.load_state_dict(checkpoint["transformer"])

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

TEST_MDPS = None

node_instances = {}
car_instances = {}
if TEST_MDPS is None:
    node_instances = {
        value: [
            build_instance(
                rng,
                value,
                comparison["node_sweep"]["fixed_cars"],
                comparison,
            )
            for _trial in range(trials)
        ]
        for value in node_values
    }
    car_instances = {
        value: [
            build_instance(
                rng,
                comparison["car_sweep"]["fixed_nodes"],
                value,
                comparison,
            )
            for _trial in range(trials)
        ]
        for value in car_values
    }
else:

    node_instances = TEST_MDPS.get("nodes", {})
    car_instances = TEST_MDPS.get("cars", {})

for sweep_name, values, instances, output_file, x_label in [
    (
        "nodes",
        node_values,
        node_instances,
        comparison["node_sweep"]["output_file"],
        "Number of nodes",
    ),
    (
        "cars",
        car_values,
        car_instances,
        comparison["car_sweep"]["output_file"],
        "Number of cars",
    ),
]:
    heuristic_means = [
        np.mean([distance for _mdp, distance in instances[value]])
        for value in values
    ]
    plt.figure(
        figsize=(
            comparison["plot"]["width"],
            comparison["plot"]["height"],
        )
    )
    plt.plot(values, heuristic_means, marker="o", label="Clarke-Wright")

    heuristic_all = np.asarray(
        [
            distance
            for value in values
            for _mdp, distance in instances[value]
        ]
    )
    for name, model in models.items():
        model_means = []
        model_all = []
        for value in values:
            distances = []
            for mdp, _heuristic_distance in instances[value]:
                model.mdp = mdp
                model.routes = [model.new_route() for _ in range(mdp.num_cars)]
                model.used = {mdp.depot_num}
                model.run_episode(train=False)

                routes = model.routes
                customers = sorted(
                    node
                    for route in routes
                    for node in route["path"]
                    if node != mdp.depot_num
                )
                if customers != list(range(1, mdp.num_nodes)):
                    raise RuntimeError(f"{name} returned an invalid solution")
                distances.append(float(sum(route["total_distance"] for route in routes)))
            model_means.append(np.mean(distances))
            model_all.extend(distances)

        model_all = np.asarray(model_all)
        gaps = 100.0 * (model_all - heuristic_all) / heuristic_all

        plt.plot(values, model_means, marker="o", label=f"{name}")

    plt.xlabel(x_label)
    plt.ylabel("Mean total distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_file,
        dpi=comparison["plot"]["dpi"],
    )
    plt.close()
