
import random

import numpy as np
import torch
import yaml

from test_samples import generate_test_samples, save_test_samples

with open("conf.yaml", "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

comparison = config["comparison"]
random.seed(comparison["seed"])
np.random.seed(comparison["seed"])
torch.manual_seed(comparison["seed"])

test_samples = generate_test_samples(comparison)

output_path = comparison["test_samples_file"]
save_test_samples(test_samples, output_path)

num_node_instances = sum(len(v) for v in test_samples["nodes"].values())
num_car_instances = sum(len(v) for v in test_samples["cars"].values())
print(
    f"Saved {num_node_instances} node-sweep instances and "
    f"{num_car_instances} car-sweep instances to {output_path}"
)
