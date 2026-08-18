import yaml
import torch
import matplotlib.pyplot as plt
from MDP import MDP
from heuristic import ClarkeWrightSavings
from onlyA2C.only_A2C import OnlyA2C
from transformerA2C.transformer_A2C import TransformerA2C


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


with open("conf.yaml", "r") as f:
    config = yaml.safe_load(f)

comp = config.get("comparison", {})
ckpts = comp.get("checkpoints", {})

only = OnlyA2C(
    actor_input_dim=OnlyA2C.INPUT_DIM,
    hidden_dim=config["hidden_dim"],
    lr=config["lr"],
    discount=config["discount"],
    entropy_coef=0.01,
    value_loss_coef=0.5,
    episodes_per_update=5,
    num_epoches=config["num_epoches"],
    max_num_nodes=config["max_num_nodes"],
    min_num_nodes=config["min_num_nodes"],
    max_num_cars=config["max_num_cars"],
    min_num_cars=config["min_num_cars"],
    cars_capacity=config["cars_capacity"],
    depot_num=config["depot_num"],
    min_distance=config["min_distance"],
    max_distance=config["max_distance"],
    min_node_dem=config["min_node_dem"],
    max_node_dem=config["max_node_dem"],
    max_grad_norm=config["max_grad_norm"],
)

trans_conf = comp.get("transformer_a2c", {})
trans = TransformerA2C(
    num_layers=trans_conf.get("num_layers", 2),
    num_heads=trans_conf.get("num_heads", 1),
    model_dim=trans_conf.get("model_dim", 5),
    FF_hidden_dim=trans_conf.get("feed_forward_dim", 16),
    actor_input_dim=3 * trans_conf.get("model_dim", 5) + 5,
    critic_input_dim=2 * trans_conf.get("model_dim", 5) + 3,
    hidden_dim=config["hidden_dim"],
    lr=config["lr"],
    discount=config["discount"],
    entropy_coef=0.01,
    value_loss_coef=0.5,
    episodes_per_update=5,
    num_epoches=config["num_epoches"],
    max_num_nodes=config["max_num_nodes"],
    min_num_nodes=config["min_num_nodes"],
    max_num_cars=config["max_num_cars"],
    min_num_cars=config["min_num_cars"],
    cars_capacity=config["cars_capacity"],
    depot_num=config["depot_num"],
    min_distance=config["min_distance"],
    max_distance=config["max_distance"],
    min_node_dem=config["min_node_dem"],
    max_node_dem=config["max_node_dem"],
    max_grad_norm=config["max_grad_norm"],
)

try:
    from GNNA2C.gnn_A2C import GNNA2C
    gnn_defaults = comp.get("gnn_dqn", {})
    gnn_conf = comp.get("gnn_a2c", {})
    gnn_input_dim = gnn_conf.get("input_dim", gnn_defaults.get("input_dim", gnn_defaults.get("input_dim", 3)))
    gnn_hidden = gnn_conf.get("hidden_dim", gnn_defaults.get("hidden_dim", 16))
    gnn_output = gnn_conf.get("gnn_output_dim", gnn_defaults.get("embedding_dim", gnn_defaults.get("gnn_output_dim", 8)))
    gnn = GNNA2C(
        input_dim_gnn=gnn_input_dim,
        gnn_hidden_dim=gnn_hidden,
        gnn_output_dim=gnn_output,
        large_value=gnn_conf.get("large_value", gnn_defaults.get("large_value", 100)),
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        discount=config["discount"],
        entropy_coef=0.01,
        value_loss_coef=0.5,
        episodes_per_update=5,
        num_epoches=config["num_epoches"],
        max_num_nodes=config["max_num_nodes"],
        min_num_nodes=config["min_num_nodes"],
        max_num_cars=config["max_num_cars"],
        min_num_cars=config["min_num_cars"],
        cars_capacity=config["cars_capacity"],
        depot_num=config["depot_num"],
        min_distance=config["min_distance"],
        max_distance=config["max_distance"],
        min_node_dem=config["min_node_dem"],
        max_node_dem=config["max_node_dem"],
        max_grad_norm=config["max_grad_norm"],
    )
except Exception:
    gnn = None

try:
    from gnntransA2C.gnntransA2C import GNNTransA2C
    gtn_defaults = comp.get("gnn_transformer_dqn", {})
    gt_conf = comp.get("gnn_transformer_a2c", {})
    gtn = GNNTransA2C(
        input_dim_gnn=gt_conf.get("gnn_input_dim", gtn_defaults.get("gnn_input_dim", 3)),
        gnn_hidden_dim=gt_conf.get("gnn_hidden_dim", gtn_defaults.get("gnn_hidden_dim", 16)),
        gnn_output_dim=gt_conf.get("gnn_output_dim", gtn_defaults.get("gnn_output_dim", gtn_defaults.get("gnn_output_dim", 8))),
        num_layers=gt_conf.get("num_layers", gtn_defaults.get("num_layers", 3)),
        large_value=gt_conf.get("large_value", gtn_defaults.get("large_value", 100)),
        num_heads=gt_conf.get("num_heads", gtn_defaults.get("num_heads", 2)),
        model_dim=gt_conf.get("model_dim", gtn_defaults.get("model_dim", 8)),
        FF_hidden_dim=gt_conf.get("feed_forward_dim", gtn_defaults.get("feed_forward_dim", 24)),
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        discount=config["discount"],
        entropy_coef=0.01,
        value_loss_coef=0.5,
        episodes_per_update=5,
        num_epoches=config["num_epoches"],
        max_num_nodes=config["max_num_nodes"],
        min_num_nodes=config["min_num_nodes"],
        max_num_cars=config["max_num_cars"],
        min_num_cars=config["min_num_cars"],
        cars_capacity=config["cars_capacity"],
        depot_num=config["depot_num"],
        min_distance=config["min_distance"],
        max_distance=config["max_distance"],
        min_node_dem=config["min_node_dem"],
        max_node_dem=config["max_node_dem"],
        max_grad_norm=config["max_grad_norm"],
    )
except Exception:
    gtn = None

models = {"heuristic": None, "OnlyA2C": only, "TransformerA2C": trans, "GNNA2C": gnn, "GNNTransA2C": gtn}

for name, model in models.items():
    if name == "heuristic" or model is None:
        continue
    print(f"Training {name}")
    model.A2C_train()
    ck = {"actor": model.actor.state_dict(), "critic": model.critic.state_dict()}
    if hasattr(model, "gnn") and model.gnn is not None:
        ck["gnn"] = model.gnn.state_dict()
    if hasattr(model, "transforemer") and model.transforemer is not None:
        ck["transformer"] = model.transforemer.state_dict()
    key_map = {"OnlyA2C": "only_a2c", "TransformerA2C": "transformer_a2c", "GNNA2C": "gnn_a2c", "GNNTransA2C": "gnn_transformer_a2c"}
    path = ckpts.get(key_map.get(name, ""))
    if path:
        torch.save(ck, path)

results = {n: [] for n in models.keys()}
x_values = []
for n_nodes in range(config["min_num_nodes"], min(config["max_num_nodes"] + 1, config["min_num_nodes"] + 5)):
    for n_cars in range(config["min_num_cars"], config["max_num_cars"] + 1):
        test_mdp = MDP(n_nodes, config["depot_num"], n_cars, config["cars_capacity"])
        test_mdp.fill_distance_matrix(config["min_distance"], config["max_distance"])
        test_mdp.fill_node_dem_matrix(config["min_node_dem"], config["max_node_dem"])

        cws = ClarkeWrightSavings(test_mdp)
        feasible = cws.CWS_solve()
        if not feasible:
            continue

        heuristic_trial_dis = total_dis(cws.list_routes, test_mdp)
        results["heuristic"].append(heuristic_trial_dis)
        x_values.append(n_nodes)

        for name, model in models.items():
            if name == "heuristic" or model is None:
                continue
            model.mdp = test_mdp
            model.routes = [model.new_route() for _ in range(test_mdp.num_cars)]
            model.used = {test_mdp.depot_num}
            model.run_episode(train=False)
            total_distance = sum(route["total_distance"] for route in model.routes)
            results[name].append(float(total_distance))

for name, vals in results.items():
    if not vals:
        continue
    plt.plot(x_values, vals, label=name)
xticks = sorted(set(x_values))
plt.xlabel("num nodes")
plt.xticks(xticks)
plt.legend()
plt.savefig("compare_a2c.png")
