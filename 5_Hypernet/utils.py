import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.reference_direction import UniformReferenceDirectionFactory


def get_ref_dirs(n_obj):
    if n_obj == 2:
        ref_dirs = UniformReferenceDirectionFactory(2, n_points=25).do()
    elif n_obj == 3:
        #ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=15).do()
        ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=100).do()
    else:
        raise Exception("Please provide reference directions for more than 3 objectives!")
    return ref_dirs

def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]

def plot_trajectories(obs_true=None, obs_predict=None, obs_pareto_front=None, title = None, save=None, figsize=(16, 8)):
    plt.figure(figsize=figsize)
    ig, ax = plt.subplots()
    ax.scatter(
        obs_pareto_front[:, 0], obs_pareto_front[:, 1], s=20, c="gray", label="PF", marker="X"
    )
    ax.scatter(
        obs_true[:, 0], obs_true[:, 1], s=20, c="yellow", label="PF_true", marker="D"
    )
    ax.scatter(
        obs_predict[:, 0],
        obs_predict[:, 1],
        s=20,
        c="red",
        label="PF_predict",
        marker="X",
    )
    plt.legend()
    ax.set_title(title)
    if save:
        plt.savefig(save_path)
    plt.show()
    
def plot_solution_space(target_space, target_space_full, target_space_cir):
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, sharey=True)

    axes[0].scatter(target_space_full[:, 0], target_space_full[:, 1], s=10, c="red", label="X - Constraints")
    axes[0].legend()

    axes[0].scatter(target_space_cir[:, 0], target_space_cir[:, 1], s=10, c="yellow", label="Circle - (f - constraint)")
    axes[0].legend()

    axes[1].scatter(target_space[:, 0], target_space[:, 1], s=10, c="gray", label="Feasible solutions space")
    axes[1].legend()

    # Đảm bảo tỷ lệ trục bằng nhau
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()
    