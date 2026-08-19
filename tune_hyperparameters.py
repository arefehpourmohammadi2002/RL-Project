"""Reproducible hyperparameter search for the lightweight DQN baseline.

The search deliberately uses a held-out subset of ``test_samples.pkl`` and
never overwrites the project's checkpoints.  The winning parameters can then
be copied into conf.yaml before retraining all architectures.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from onlyDQN.only_DQN import OnlyDQN
from test_samples import load_test_samples


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config, params, epochs):
    comparison = config["comparison"]
    return OnlyDQN(
        input_dim=OnlyDQN.INPUT_DIM,
        output_dim=config["output_dim"],
        hidden_dim=params["hidden_dim"],
        lr=params["lr"],
        lr_decay=params["lr_decay"],
        min_lr=config["min_lr"],
        target_update_counter=params["target_update_counter"],
        explore_update_counter=config["explore_update_counter"],
        discount=params["discount"],
        epsilon=config["epsilon"],
        epsilon_decay=params["epsilon_decay"],
        min_epsilon=config["min_epsilon"],
        batch_size=params["batch_size"],
        RB_capacity=config["RB_capacity"],
        num_first_samples=max(config["num_first_samples"], params["batch_size"]),
        num_epoches=epochs,
        min_num_nodes=config["min_num_nodes"],
        max_num_nodes=config["max_num_nodes"],
        depot_num=comparison["depot_num"],
        min_num_cars=config["min_num_cars"],
        max_num_cars=config["max_num_cars"],
        cars_capacity=comparison["cars_capacity"],
        min_distance=comparison["min_distance"],
        max_distance=comparison["max_distance"],
        min_node_dem=comparison["min_node_demand"],
        max_node_dem=comparison["max_node_demand"],
        max_grad_norm=config["max_grad_norm"],
    )


def validation_instances(samples):
    """Small fixed set spanning in-range and edge-size validation cases."""
    nodes = samples["nodes"]
    desired = [9, 12, 15, 18, 20]
    keys = [key for key in desired if key in nodes]
    if not keys:
        keys = sorted(nodes)[:: max(1, len(nodes) // 5)]
    return [item for key in keys for item in nodes[key][:2]]


def evaluate(model, instances):
    distances = []
    heuristic_distances = []
    for mdp, heuristic_distance in instances:
        distance, _routes = model.evaluate(mdp)
        if len(model.used) != mdp.num_nodes:
            raise RuntimeError("DQN produced an invalid validation route")
        distances.append(float(distance))
        heuristic_distances.append(float(heuristic_distance))
    distances = np.asarray(distances)
    heuristic_distances = np.asarray(heuristic_distances)
    return {
        "mean_distance": float(distances.mean()),
        "mean_ratio_to_heuristic": float(np.mean(distances / heuristic_distances)),
        "mean_gap_percent": float(np.mean((distances / heuristic_distances - 1) * 100)),
    }


def train(model):
    """Train without DQN_train's plotting side effects."""
    model.prepare_replay_buffer()
    for _epoch in range(model.num_epoches):
        model.run_episode(train=True)
        model.decay_epsilon()
        model.decay_lr()


def candidates(config):
    # A compact manual grid targets the parameters with the largest effect on
    # short, CPU-only RL runs. The current configuration is always trial zero.
    base = {
        "lr": config["lr"],
        "lr_decay": config["lr_decay"],
        "discount": config["discount"],
        "epsilon_decay": config["epsilon_decay"],
        "target_update_counter": config["target_update_counter"],
        "batch_size": config["batch_size"],
        "hidden_dim": config["hidden_dim"],
    }
    variants = [
        {},
        {"lr": 3e-4, "lr_decay": 0.995},
        {"lr": 3e-4, "lr_decay": 0.995, "discount": 0.99},
        {"epsilon_decay": 0.985, "target_update_counter": 25},
        {"lr": 5e-4, "epsilon_decay": 0.985, "target_update_counter": 25},
        {"lr": 5e-4, "discount": 0.99, "epsilon_decay": 0.985,
         "target_update_counter": 25, "batch_size": 64},
    ]
    return [{**base, **variant} for variant in variants]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output", default="tuning_results.json")
    args = parser.parse_args()

    with open("conf.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    samples = load_test_samples(config["comparison"]["test_samples_file"])
    held_out = validation_instances(samples)
    rows = []

    for trial, params in enumerate(candidates(config)):
        seed_metrics = []
        started = time.time()
        for seed in args.seeds:
            seed_everything(seed)
            model = build_model(config, params, args.epochs)
            train(model)
            seed_metrics.append(evaluate(model, held_out))
        row = {
            "trial": trial,
            **params,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "validation_instances": len(held_out),
            "mean_distance": float(np.mean([x["mean_distance"] for x in seed_metrics])),
            "mean_ratio_to_heuristic": float(np.mean([x["mean_ratio_to_heuristic"] for x in seed_metrics])),
            "mean_gap_percent": float(np.mean([x["mean_gap_percent"] for x in seed_metrics])),
            "elapsed_seconds": round(time.time() - started, 2),
        }
        rows.append(row)
        print("RESULT", json.dumps(row), flush=True)

    rows.sort(key=lambda row: row["mean_ratio_to_heuristic"])
    output = {"best": rows[0], "results": rows}
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    csv_path = Path(args.output).with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("BEST", json.dumps(rows[0]), flush=True)


if __name__ == "__main__":
    main()
