import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from A2C.factory import CHECKPOINT_KEYS, build_models, load_config
from test_samples import load_test_samples


def compare_a2c():
    config = load_config()
    comparison = config["comparison"]
    random.seed(comparison["seed"])
    np.random.seed(comparison["seed"])
    torch.manual_seed(comparison["seed"])
    models = build_models(config)
    for name, model in models.items():
        path = comparison["checkpoints"][CHECKPOINT_KEYS[name]]
        model.load_checkpoint(torch.load(path, map_location=model.device, weights_only=True))

    samples = load_test_samples(comparison["test_samples_file"])

    def plot(instances, output_file, x_label):
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        values = sorted(instances)
        plt.figure(figsize=(comparison["plot"]["width"], comparison["plot"]["height"]))
        plt.plot(values, [np.mean([distance for _, distance in instances[x]])
                          for x in values], marker="o", label="Clarke-Wright")
        for name, model in models.items():
            means = []
            for value in values:
                distances = []
                for mdp, _ in instances[value]:
                    distance, routes = model.evaluate(mdp)
                    customers = sorted(node for route in routes for node in route["path"]
                                       if node != mdp.depot_num)
                    if customers != list(range(1, mdp.num_nodes)):
                        raise RuntimeError(f"{name} returned an invalid solution")
                    distances.append(distance)
                means.append(np.mean(distances))
            plt.plot(values, means, marker="o", label=name)
        plt.xlabel(x_label)
        plt.ylabel("Mean total distance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=comparison["plot"]["dpi"])
        plt.close()
        print(f"Created plot: {output_path.resolve()}")

    plot(samples["nodes"], comparison["a2c_output"]["node_sweep_file"],
         "Number of nodes")
    plot(samples["cars"], comparison["a2c_output"]["car_sweep_file"],
         "Number of cars")


if __name__ == "__main__":
    compare_a2c()
