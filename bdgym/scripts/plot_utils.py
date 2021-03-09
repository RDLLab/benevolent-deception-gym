"""Utility functions, etc for plotting results """
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Plot constants
LINE_WIDTH = 1.5
FONTSIZE = 12
BAR_WIDTH_LIMIT = 0.8
BAR_GAP = 0.02


def import_results(results_file: str,
                   sep: Optional[str] = None) -> pd.DataFrame:
    """Import results from file into Pandas DataFrame """
    if sep is None:
        sep = "\t"
    return pd.read_csv(results_file, sep=sep)


def plot_xy_err(ax: plt.Axes,
                df: pd.DataFrame,
                x_key: str,
                y_key: str,
                yerr_key: str,
                **kwargs):
    """Plot Line plot with shaded errors """
    x_label = kwargs.get("x_label", x_key)
    y_label = kwargs.get("y_label", y_key)
    y_lims = kwargs.get("y_lims", None)
    x_lims = kwargs.get("x_lims", None)
    alpha = kwargs.get("alpha", 0.5)
    line_label = kwargs.get("line_label", None)

    x = df[x_key]
    y = df[y_key]
    y_err = df[yerr_key]

    line = ax.plot(x, y, label=line_label, lw=LINE_WIDTH)[0]
    ax.fill_between(x, y-y_err, y+y_err, alpha)

    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.set_ylabel(y_label, fontsize=FONTSIZE)
    if y_lims:
        ax.set_ylim(y_lims)
    if x_lims:
        ax.set_xlim(x_lims)
    return line


def plot_metric_bar(ax: plt.Axes,
                    df: pd.DataFrame,
                    x_key: str,
                    y_key: str,
                    z_key: str,
                    **kwargs):
    """Plot bar plot """
    x_labels = kwargs.get("x_labels", None)
    y_label = kwargs.get("y_label", y_key)
    y_lims = kwargs.get("y_lims", None)
    z_labels = kwargs.get("z_labels", None)
    label_map = kwargs.get("label_map", {})
    yerr_key = kwargs.get("yerr_key", None)
    styles = kwargs.get("styles", None)

    if label_map is None:
        label_map = {}
    if x_labels is None:
        x_labels = df[x_key].unique().tolist()
    x_start = np.array(range(len(x_labels)))

    if z_labels is None:
        z_labels = df[z_key].unique()
    width = BAR_WIDTH_LIMIT / len(z_labels) - BAR_GAP

    bars = []
    yerr = None

    for i, z_label in enumerate(z_labels):
        z_df = df[df[z_key] == z_label]
        y = z_df[y_key]
        x = x_start + (i * width)

        # Need to ensure results are in correct order
        zx_pos = []
        for zx in z_df[x_key].unique():
            zx_idx = x_labels.index(zx)
            zx_pos.append(x[zx_idx])
        x = zx_pos

        if yerr_key is not None:
            yerr = [
                z_df[yerr_key],
                z_df[yerr_key]
            ]
        args = [x, y, width]
        kwargs = dict(
            yerr=yerr,
            align='edge',
            label=label_map.get(z_label, z_label)
        )
        if styles is None:
            bar_obj = ax.bar(*args, **kwargs)
        else:
            bar_obj = ax.bar(*args, **kwargs, **next(styles))

        bars.append(bar_obj)

    ax.set_ylabel(y_label, fontsize=FONTSIZE)
    ax.set_xticks(x_start + BAR_WIDTH_LIMIT / 2)
    x_labels = [
        label_map.get(label, label) for label in df[x_key].unique()
    ]
    ax.set_xticklabels(x_labels, fontsize=FONTSIZE)
    if y_lims:
        ax.set_ylim(y_lims)

    return bars, [label_map.get(label, label) for label in z_labels]
