import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def performance_comparison(
    heuristic_perf: np.ndarray,
    dqn_perf: np.ndarray,
    start_nodes: int = 5,
    base_filename: str = "performance",
):
    num_rows, num_cols = heuristic_perf.shape

    cars_axis = np.arange(1, num_cols + 1)
    nodes_axis = np.arange(start_nodes, start_nodes + num_rows)

    X, Y = np.meshgrid(cars_axis, nodes_axis)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf_heuristic = ax.plot_surface(
        X,
        Y,
        heuristic_perf,
        color="skyblue",
        alpha=0.6,
        edgecolor="navy",
        linewidth=0.5,
    )

    surf_dqn = ax.plot_surface(
        X,
        Y,
        dqn_perf,
        color="orange",
        alpha=0.6,
        edgecolor="darkred",
        linewidth=0.5,
    )

    ax.set_xlabel("Number of Cars", fontsize=11, labelpad=10)
    ax.set_ylabel("Number of Nodes", fontsize=11, labelpad=10)
    ax.set_zlabel("Performance", fontsize=11, labelpad=10)
    ax.set_title(
        "3D Performance Comparison: Heuristic vs DQN", fontsize=14, pad=15
    )

    surf_heuristic._facecolors2d = surf_heuristic._facecolor3d
    surf_heuristic._edgecolors2d = surf_heuristic._edgecolor3d
    surf_dqn._facecolors2d = surf_dqn._facecolor3d
    surf_dqn._edgecolors2d = surf_dqn._edgecolor3d

    ax.legend(
        [surf_heuristic, surf_dqn], ["Heuristic", "DQN"], loc="upper left"
    )
    ax.view_init(elev=25, azim=135)

    plt.tight_layout()
    plt.savefig(
        f"{base_filename}_3d.jpg", format="jpg", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    heuristic_by_nodes = np.mean(heuristic_perf, axis=1)
    dqn_by_nodes = np.mean(dqn_perf, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        nodes_axis,
        heuristic_by_nodes,
        label="Heuristic",
        color="navy",
        marker="o",
    )
    ax.plot(nodes_axis, dqn_by_nodes, label="DQN", color="darkred", marker="s")
    ax.set_xlabel("Number of Nodes", fontsize=11)
    ax.set_ylabel("Average Performance", fontsize=11)
    ax.set_title("Performance Comparison by Number of Nodes", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        f"{base_filename}_by_nodes.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    heuristic_by_cars = np.mean(heuristic_perf, axis=0)
    dqn_by_cars = np.mean(dqn_perf, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        cars_axis,
        heuristic_by_cars,
        label="Heuristic",
        color="navy",
        marker="o",
    )
    ax.plot(cars_axis, dqn_by_cars, label="DQN", color="darkred", marker="s")
    ax.set_xlabel("Number of Cars", fontsize=11)
    ax.set_ylabel("Average Performance", fontsize=11)
    ax.set_title("Performance Comparison by Number of Cars", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        f"{base_filename}_by_cars.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)