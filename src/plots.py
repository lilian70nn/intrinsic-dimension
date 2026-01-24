"""
### Plotting utilities for ID analysis
* Color/marker helpers by model family
* `plot_fig3b`: layer-wise ID vs relative depth for multiple models
* `plot_fig9A/B/C`: ID dynamics over epochs and vs error
* `plot_fig5c`: trained vs untrained ID / PC-ID / ED across depth
"""

import re
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize



def title_to_filename(title, ext="png"):
    """
    Convert a plot title into a filesystem-safe filename while preserving semantics.

    """
    s = title.lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")

    return f"{s}.{ext}"

def family_color(name: str):
    n = name.lower()
    if "alex" in n:   return "#f2a23a"
    if "vgg" in n and "bn" in n: return "k"
    if "vgg" in n:    return "#1f77b4"
    if "resnet" in n: return "#2ca02c"

    rng = random.Random(n)

    r = rng.randint(0, 200)
    g = rng.randint(0, 200)
    b = rng.randint(0, 200)

    return '#%02X%02X%02X' % (r, g, b)

def model_depth_number(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 10


def family_key(name: str):
    n = name.lower()
    if "resnet" in n: return "resnet"
    if "vgg" in n and "bn" in n: return "vgg_bn"
    if "vgg" in n: return "vgg"
    if "alex" in n: return "alexnet"
    return "other"

def build_family_stats(data):
    depths = defaultdict(list)
    for model_name,model_measures in data.items():
        depths[family_key(model_name)].append(model_depth_number(model_name))
    stats = {}
    for k, vals in depths.items():
        dmin, dmax = min(vals), max(vals)
        stats[k] = (dmin, dmax)
    return stats

def marker_size_for_model_family(name, family_stats, min_s=30, max_s=90):
    fam = family_key(name)
    d = model_depth_number(name)
    dmin, dmax = family_stats[fam]
    if dmax == dmin:
        return (min_s + max_s) / 2
    frac = (d - dmin) / (dmax - dmin)
    s = min_s + (max_s - min_s) * frac
    return float(np.clip(s, min_s, max_s))


def plot_fig3b(data,estimator_name,title,annotate=False,show=True,savepath=None):

    """
    Plot layer-wise intrinsic dimension curves for multiple models (reproducing Fig. 3B).

    Args:
        data (dict):
            Output of compute_id_dynamics_across_models().
            For each model:
                - data[model_name]["depths"]: list of layer depth indices
                - data[model_name][estimator_name]: list of ID values (one per layer)
        estimator_name (str): Key selecting which ID estimator to plot ("TwoNN", "MLE", etc.).
        title (str): Plot title (also used to derive a safe filename when saving).

    Note:
        `data[model_name][estimator_name]` **is the list of layer-wise ID values**.
        This function only visualizes precomputed results.
    """

    plt.figure(figsize=(9,6))
    legend_handles = []
    legend_labels  = []

    family_stats = build_family_stats(data)
    for model_name,model_measures in data.items():
        layers = np.asarray(model_measures["depths"], dtype=float)
        ids = np.asarray(model_measures[estimator_name], dtype=float)

        depth_names = model_measures.get("depth_names", None)

        max_layer = max(layers.max(), 1.0)
        x = layers / max_layer
        y = ids


        color = family_color(model_name)
        s = marker_size_for_model_family(model_name,family_stats)


        plt.plot(x, y, "-", lw=2, color=color, alpha=0.9, label=model_name)
        plt.scatter(x, y, s=s, color=color, edgecolors="white", linewidths=1.2, zorder=3)

        dx = -0.05
        dy = 0.05
        if annotate and depth_names is not None:
            for xi, yi, name in zip(x, y, depth_names):
                plt.text(
                    xi + dx, yi + dy,
                    name,
                    color=color,
                    fontsize=7,
                    alpha=0.9
                )

        handle = Line2D([0], [0], color=color, marker='o', linestyle='-',
                        markerfacecolor=color, markeredgecolor='white',
                        linewidth=2, markersize=np.sqrt(s),
                        label=model_name)
        legend_handles.append(handle)
        legend_labels.append(model_name)


    plt.xlabel("relative depth", fontsize=13)
    plt.ylabel("ID", fontsize=13)
    plt.title(title, fontsize=14)

    plt.xlim(-0.02, 1.02)
    #plt.ylim(0, 160)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


    uniq = dict(zip(legend_labels, legend_handles))
    plt.legend(uniq.values(), uniq.keys(), ncol=2, frameon=False, fontsize=11,bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if savepath is not None:
        outdir = Path(savepath)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / title_to_filename(title)
        plt.savefig(outfile, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(plt.gcf())






def plot_fig9A(epoch_records, depths, title, annotate=False, show=True, savepath=None):


    """
    Plot the evolution of layer-wise intrinsic dimension across epochs (Fig. 9A reproduction).

    Args:
        epoch_records (list):
            List where each element corresponds to one epoch.
            Each entry has the format:
                [epoch_index, epoch_acc, id_values]
            - epoch_index (int): epoch number
            - epoch_acc (float or None): test accuracy for that epoch
            - id_values (list):
                  Layer-wise intrinsic dimension values, each entry is a scalar.

        depths (array-like):
            Depth index for each layer (e.g., 0, 1, 2, ...); used to compute relative depth.

    Notes:
        - `epoch_records[0]` is interpreted as the UNTRAINED network.
        - For each epoch, only the ID mean is plotted.
        - Color encodes epoch progression (Viridis colormap).
        - X-axis uses relative depth = depth / max(depth).

    This function visualizes precomputed ID dynamics, but does not compute ID.
    """


    num_epochs = len(epoch_records)
    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=num_epochs - 1)

    fig, ax = plt.subplots(figsize=(8,6))

    rel_depth = depths / depths[-1]

    epoch0_id = epoch_records[0][2]
    ax.plot(rel_depth, epoch0_id, color="black", linewidth=3, label="UNTRAINED")

    for i in range(1, num_epochs):

        epoch_idx, _, id_values = epoch_records[i]
        color = cmap(norm(i))

        ax.plot(rel_depth, id_values, color=color, linewidth=1.3, alpha=0.9)

        if i <= 3:
            ax.text(rel_depth[1], id_values[1],
                    f"EPOCH {epoch_idx}",
                    color=color, fontsize=8, rotation=25)

        if i == num_epochs - 1:
            ax.text(rel_depth[1], id_values[1],
                    f"EPOCH {epoch_idx}",
                    color=color, fontsize=8, rotation=25)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("epoch")

    ax.set_xlabel("relative depth")
    ax.set_ylabel("ID")
    #ax.set_ylim(0, 80)

    if annotate:
        depth_names = epoch_records[0][-1]
        y_top = ax.get_ylim()[1]

        for i in range(len(depth_names) - 1):
            x_l = rel_depth[i]
            x_r = rel_depth[i + 1]
            x_mid = 0.5 * (x_l + x_r)

            stage_label = depth_names[i + 1]

            ax.text(
                x_mid,
                y_top*1.01,
                stage_label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                alpha=0.9,
            )

            ax.axvline(
                x=x_r,
                color="k",
                lw=0.8,
                ls="--",
                alpha=0.25,
            )

    ax.legend()
    ax.set_title(title,pad=30)
    plt.tight_layout()

    if savepath is not None:
        outdir = Path(savepath)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / title_to_filename(title)
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_fig9B(batch_records,title,show=True,savepath=None):

    """
    Plot intrinsic dimension of the last hidden layer together with training/test
    accuracy across iterations (reproduction of Fig. 9B).

    Args:
        batch_records (list):
            Each item corresponds to one training mini-batch and has the form:
                [iteration, train_acc, test_acc, last_id]

    Notes:
        - ID values are plotted on the left y-axis.
        - Training and test accuracy are plotted on the right y-axis.
        - This function only visualizes precomputed quantities; it does not compute ID itself.
    """
    batch_records = np.array(batch_records, dtype=object)

    iters = batch_records[:,0].astype(int)
    train_acc = batch_records[:,1].astype(float)
    test_acc  = batch_records[:,2]
    id_vals   = batch_records[:,3]


    mask_id = np.array([v is not None for v in id_vals])
    id_iters = iters[mask_id]
    id_vals  = id_vals[mask_id].astype(float)


    mask_test = np.array([v is not None for v in test_acc])
    test_iters = iters[mask_test]
    test_acc   = test_acc[mask_test].astype(float)

    train_acc_pct = train_acc * 100.0
    test_acc_pct  = test_acc * 100.0

    fig, ax1 = plt.subplots(figsize=(8,6))

    ax1.plot(id_iters, id_vals, 'o-', color='black', label="ID last hidden layer")
    ax1.set_xlabel("iterations (n. of mini-batches)")
    ax1.set_ylabel("ID")
    ax1.set_ylim(0, max(id_vals)*1.2)

    ax2 = ax1.twinx()
    ax2.plot(iters, train_acc_pct, linestyle='--', color='red', alpha=0.6, label="training accuracy")
    ax2.plot(test_iters, test_acc_pct, linestyle='-', color='blue', label="test accuracy")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(title)
    plt.tight_layout()

    if savepath is not None:
        outdir = Path(savepath)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / title_to_filename(title)
        plt.savefig(outfile, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)




def plot_fig9C(epoch_records,title,show=True,savepath=None):


    """
    Plot the relationship between test error and intrinsic dimension of the last hidden
    layer across epochs (reproducing Fig. 9C).

    Args:
        epoch_records (list):
            Each entry corresponds to one epoch and has the format:
                [epoch_idx, test_acc, id_list]

            - epoch_idx (int):
                Epoch number.

            - test_acc (float):
                Test accuracy in [0, 1]. Converted internally to test error (%).

            - id_list (list of floats):
                Intrinsic-dimension values for each layer at that epoch.
                The last hidden layer is taken as id_list[-2].

    Notes:
        - Points are colored by epoch number (Viridis colormap).
        - Highlighted epochs include: 0, 1, 2, 5, 10, and the final epoch.
        - This function only visualizes precomputed ID values and does not
          perform ID estimation itself.
    """

    epochs = []
    errors = []
    ids_last = []

    for rec in epoch_records:
        epoch = rec[0]
        acc   = rec[1]
        id_list = rec[2]


        err = (1.0 - float(acc)) * 100.0

        mean_id = id_list[-2]

        epochs.append(epoch)
        errors.append(err)
        ids_last.append(float(mean_id))

    epochs = np.array(epochs, dtype=float)
    errors = np.array(errors, dtype=float)
    ids_last = np.array(ids_last, dtype=float)

    fig, ax = plt.subplots(figsize=(6,6))

    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=epochs.min(), vmax=epochs.max())

    sc = ax.scatter(errors, ids_last,
                     c=epochs, cmap=cmap, norm=norm,
                     s=20, edgecolors='none')

    ax.set_xlabel("error (%)")
    ax.set_ylabel("ID")

    ymin = max(0, ids_last.min() - 1)
    ymax = ids_last.max() + 1
    ax.set_ylim(ymin, ymax)

    cbar = plt.colorbar(sc)
    cbar.set_label("epoch")

    highlight_epochs = [0, 1, 2, 5, 10, len(epochs)-1]
    for ep in highlight_epochs:
        if ep in epochs:
            i = np.where(epochs == ep)[0][0]
            ax.scatter(errors[i], ids_last[i],
                        s=80, facecolors='none', edgecolors='lightgreen', linewidths=1.5)
            ax.text(errors[i]+1, ids_last[i],
                     f"EPOCH {int(ep)}",
                     fontsize=8, color="seagreen", rotation=40)


    ax.set_title(title)
    plt.tight_layout()

    if savepath is not None:
        outdir = Path(savepath)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / title_to_filename(title)
        plt.savefig(outfile, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)



def rescale_to_range(arr, new_min=0, new_max=400):
    arr = np.asarray(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.full_like(arr, new_min, dtype=float)
    return (arr - mn) / (mx - mn) * (new_max - new_min) + new_min


def plot_fig5c(model_name, list_epoch_0_and_epoch_last, depths,ed_rescale_max,
               title,annotate=False,show=True,savepath=None):


    """
    Plot Fig.5C-style curves:
    ID (trained/untrained), PC-ID (trained/untrained), and rescaled ED vs relative depth.

    list_epoch_0_and_epoch_last:
        [ record_epoch0, record_epoch_last ],
        where each record is:
        [epoch_idx, test_acc, ID_list, ED_list, PCA_dim_list]
    """

    d_tr = list_epoch_0_and_epoch_last[1]
    d_un = list_epoch_0_and_epoch_last[0]

    x = depths / depths.max()

    y_id_tr = np.asarray(d_tr[2], dtype=float)
    y_id_un = np.asarray(d_un[2], dtype=float)

    y_pcid_tr = np.asarray(d_tr[4], dtype=float)
    y_pcid_un = np.asarray(d_un[4], dtype=float)

    y_ed_rescaled = rescale_to_range(np.asarray(d_tr[3], dtype=float),
                                     new_min=0, new_max=ed_rescale_max)

    plt.figure(figsize=(8,5))

    # ID
    plt.plot(x, y_id_tr, "o-",  color="k",        lw=2, ms=6, label="ID trained")
    plt.plot(x, y_id_un, "o--", color="k",  alpha=0.8, lw=2, ms=6, label="ID untrained")

    # PC-ID
    plt.plot(x, y_pcid_tr, "o-",  color="#d62728",        lw=2, ms=6, label="PC-ID trained")
    plt.plot(x, y_pcid_un, "o--", color="#d62728", alpha=0.8, lw=2, ms=6, label="PC-ID untrained")

    # ED (rescaled)
    plt.plot(x, y_ed_rescaled, "-", color="#1f77b4", lw=2, label="ED (rescaled)")

    plt.xlabel("relative depth", fontsize=12)
    plt.ylabel("intrinsic dimension", fontsize=12)
    plt.xlim(-0.02, 1.02)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ymax = max(y_id_tr.max(), y_id_un.max(),
               y_pcid_tr.max(), y_pcid_un.max(),
               y_ed_rescaled.max())
    plt.ylim(0, np.ceil(ymax*1.1 / 10) * 10)

    h, l = plt.gca().get_legend_handles_labels()
    bylabel = dict(zip(l, h))
    plt.legend(bylabel.values(), bylabel.keys(),
               frameon=False, bbox_to_anchor=(1.02, 1),
               loc="upper left", fontsize=11)

    plt.title(title, fontsize=13, pad=30)

    plt.tight_layout()

    if annotate:
        depth_names = list_epoch_0_and_epoch_last[0][-1]
        x_rel = x
        ax = plt.gca()

        y_top = ax.get_ylim()[1]
        for i in range(len(depth_names) - 1):
            x_l = x_rel[i]
            x_r = x_rel[i + 1]
            x_mid = 0.5 * (x_l + x_r)

            label = depth_names[i + 1]

            ax.text(
                x_mid,
                y_top * 1.03,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                rotation=0,
            )

            ax.axvline(
                x=x_r,
                color="k",
                lw=0.8,
                ls="--",
                alpha=0.25,
            )
    if savepath is not None:
        outdir = Path(savepath)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / title_to_filename(title)
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(plt.gcf())
