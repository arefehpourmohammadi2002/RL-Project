# A2C models

This folder mirrors the project's four DQN architectures with on-policy
Advantage Actor-Critic (A2C): `OnlyA2C`, `GNNA2C`, `TransformerA2C`, and
`GNNTransformerA2C`.

From the project root, train and save every A2C checkpoint with:

```bash
python -m A2C.train_all_a2c
```

After the checkpoints exist, compare the four models on the same saved test
instances used by DQN:

```bash
python -m A2C.compare_a2c
```

The common routing logic and A2C update are in `a2c_parent.py`; architecture
features are in `models.py`; model construction is centralized in `factory.py`.
