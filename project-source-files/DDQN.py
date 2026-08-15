import itertools

import torch
import torch.nn.functional as F

from MDP import MDP

# How many different graph sizes to generate and compare. Edit freely.
NODE_COUNTS = [5, 8, 10, 15]
NUM_CARS = 4
CARS_CAPACITY = 100
MIN_DIS = 1
MAX_DIS = 10
MIN_CAP = 1
MAX_CAP = 60
DEPOT_NUM = 0


def cosine(a, b):
    return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


def build_mdp(num_nodes):
    mdp = MDP(num_nodes, DEPOT_NUM, NUM_CARS, CARS_CAPACITY)
    mdp.fill_distance_matrix(MIN_DIS, MAX_DIS)
    mdp.fill_node_cap_matrix(MIN_CAP, MAX_CAP)
    return mdp


if __name__ == "__main__":
    torch.manual_seed(0)

    mdp_list = [build_mdp(n) for n in NODE_COUNTS]

    for mdp in mdp_list:
        print(f"num_nodes={mdp.num_nodes}: distance_matrix shape={mdp.distance_matrix.shape}")

    print()
    print("=== Cosine similarity between RAW distance matrices, before any GNN processing ===")
    print("(compared over the top-left NxN submatrix both instances share, "
          "diagonal zeroed since self-distance is inf, not a meaningful value)")

    for i, j in itertools.combinations(range(len(mdp_list)), 2):
        mdp_i, mdp_j = mdp_list[i], mdp_list[j]

        d1 = torch.as_tensor(mdp_i.distance_matrix, dtype=torch.float32)
        d2 = torch.as_tensor(mdp_j.distance_matrix, dtype=torch.float32)

        # Different mdps can have different num_nodes -- only compare the
        # overlapping top-left submatrix both actually have.
        n = min(d1.size(0), d2.size(0))
        d1_sub = d1[:n, :n].clone()
        d2_sub = d2[:n, :n].clone()
        d1_sub.fill_diagonal_(0)
        d2_sub.fill_diagonal_(0)

        sim = cosine(d1_sub, d2_sub)
        print(f"  mdp[{i}] (num_nodes={mdp_i.num_nodes}) vs mdp[{j}] (num_nodes={mdp_j.num_nodes}), "
              f"compared over first {n}x{n}: {sim:.4f}")

    print()
    all_sims = []
    for i, j in itertools.combinations(range(len(mdp_list)), 2):
        d1 = torch.as_tensor(mdp_list[i].distance_matrix, dtype=torch.float32)
        d2 = torch.as_tensor(mdp_list[j].distance_matrix, dtype=torch.float32)
        n = min(d1.size(0), d2.size(0))
        d1_sub = d1[:n, :n].clone()
        d2_sub = d2[:n, :n].clone()
        d1_sub.fill_diagonal_(0)
        d2_sub.fill_diagonal_(0)
        all_sims.append(cosine(d1_sub, d2_sub))

    print(f"average cosine similarity across all pairs: {sum(all_sims)/len(all_sims):.4f}")
    print()
    print("This is the BASELINE similarity inherent to the raw random data itself --")
    print("any GNN output similarity meaningfully higher than this means the network")
    print("is amplifying correlation, not just inheriting it from the input.")