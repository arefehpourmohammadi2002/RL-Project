import random

import numpy as np
import torch

from A2C.factory import CHECKPOINT_KEYS, build_models, load_config


def train_all_a2c():
    config = load_config()
    seed = config["comparison"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for name, model in build_models(config).items():
        print(f"Training {name}")
        model.A2C_train()
        checkpoint_path = config["comparison"]["checkpoints"][CHECKPOINT_KEYS[name]]
        torch.save(model.checkpoint(), checkpoint_path)
        print(f"Saved {checkpoint_path}")


if __name__ == "__main__":
    train_all_a2c()
