import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from GNNDQN.gnn_DQN import GNNDQN
from gnntransDQN.gnntransDQN import GNNTransDQN
from onlyDQN.only_DQN import OnlyDQN
from test_samples import load_test_samples
from transformerDQN.transformer_DQN import TransformerDQN

seed = 34
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

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
        input_dim=3 * gnn_config["embedding_dim"] + 11,
        **common,
    ),
    "TransformerDQN": TransformerDQN(
        num_layers=transformer_config["num_layers"],
        num_heads=transformer_config["num_heads"],
        model_dim=transformer_config["model_dim"],
        FF_hidden_dim=transformer_config["feed_forward_dim"],
        input_dim=3 * transformer_config["model_dim"] + 11,
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
        input_dim=3 * hybrid_config["gnn_output_dim"] + 11,
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
    if "state_net" not in checkpoint or "action_net" not in checkpoint:
        raise RuntimeError(
            f"{name} checkpoint predates the shared feature update; "
            "retrain it before comparison"
        )
    model.state_net.load_state_dict(checkpoint["state_net"])
    model.action_net.load_state_dict(checkpoint["action_net"])
    if hasattr(model, "gnn") and "gnn" in checkpoint:
        model.gnn.load_state_dict(checkpoint["gnn"])
    if hasattr(model, "transforemer") and "transformer" in checkpoint:
        model.transforemer.load_state_dict(checkpoint["transformer"])

test_samples = load_test_samples(comparison["test_samples_file"])
node_instances = test_samples["nodes"]
car_instances = test_samples["cars"]
node_values = sorted(node_instances)
car_values = sorted(car_instances)

def mean_distance(instances, value):
    distances = [distance for _mdp, distance in instances[value]]
    return np.mean(distances)


def run_model_on_instance(name, model, mdp):
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
    return float(sum(route["total_distance"] for route in routes))


def plot_sweep(values, instances, output_file, x_label):
    plt.figure(figsize=(comparison["plot"]["width"], comparison["plot"]["height"]))

    heuristic_means = [mean_distance(instances, value) for value in values]
    plt.plot(values, heuristic_means, marker="o", label="Clarke-Wright")

    for name, model in models.items():
        model_means = []
        for value in values:
            distances = [
                run_model_on_instance(name, model, mdp)
                for mdp, _heuristic_distance in instances[value]
            ]
            model_means.append(np.mean(distances))
        plt.plot(values, model_means, marker="o", label=name)

    plt.xlabel(x_label)
    plt.ylabel("Mean total distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=comparison["plot"]["dpi"])
    plt.close()


plot_sweep(node_values, node_instances, comparison["node_sweep"]["output_file"], "Number of nodes")
plot_sweep(car_values, car_instances, comparison["car_sweep"]["output_file"], "Number of cars")
