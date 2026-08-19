import yaml

from A2C.models import GNNA2C, GNNTransformerA2C, OnlyA2C, TransformerA2C


def load_config(path="conf.yaml"):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def build_models(config):
    comparison = config["comparison"]
    a2c = config.get("a2c", {})
    common = {
        "output_dim": config["output_dim"],
        "hidden_dim": comparison["hidden_dim"],
        "lr": a2c.get("lr", config["lr"]),
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
        "min_num_nodes": config["min_num_nodes"],
        "max_num_nodes": config["max_num_nodes"],
        "depot_num": comparison["depot_num"],
        "min_num_cars": config["min_num_cars"],
        "max_num_cars": config["max_num_cars"],
        "cars_capacity": comparison["cars_capacity"],
        "min_distance": comparison["min_distance"],
        "max_distance": comparison["max_distance"],
        "min_node_dem": comparison["min_node_demand"],
        "max_node_dem": comparison["max_node_demand"],
        "max_grad_norm": config["max_grad_norm"],
        "value_coef": a2c.get("value_coef", 0.5),
        "entropy_coef": a2c.get("entropy_coef", 0.01),
    }
    gnn = comparison["gnn_dqn"]
    transformer = comparison["transformer_dqn"]
    hybrid = comparison["gnn_transformer_dqn"]
    return {
        "OnlyA2C": OnlyA2C(input_dim=OnlyA2C.INPUT_DIM, **common),
        "GNNA2C": GNNA2C(
            input_dim_gnn=gnn["input_dim"], gnn_hidden_dim=gnn["hidden_dim"],
            gnn_output_dim=gnn["embedding_dim"] + 3,
            large_value=gnn["large_value"],
            input_dim=3 * gnn["embedding_dim"] + 5, **common),
        "TransformerA2C": TransformerA2C(
            num_layers=transformer["num_layers"], num_heads=transformer["num_heads"],
            model_dim=transformer["model_dim"],
            FF_hidden_dim=transformer["feed_forward_dim"],
            input_dim=3 * transformer["model_dim"] + 5, **common),
        "GNN-TransformerA2C": GNNTransformerA2C(
            input_dim_gnn=hybrid["gnn_input_dim"],
            gnn_hidden_dim=hybrid["gnn_hidden_dim"],
            gnn_output_dim=hybrid["gnn_output_dim"] + 3,
            large_value=hybrid["large_value"], num_layers=hybrid["num_layers"],
            num_heads=hybrid["num_heads"], model_dim=hybrid["model_dim"],
            FF_hidden_dim=hybrid["feed_forward_dim"],
            input_dim=3 * hybrid["gnn_output_dim"] + 5, **common),
    }


CHECKPOINT_KEYS = {
    "OnlyA2C": "only_a2c", "GNNA2C": "gnn_a2c",
    "TransformerA2C": "transformer_a2c",
    "GNN-TransformerA2C": "gnn_transformer_a2c",
}
