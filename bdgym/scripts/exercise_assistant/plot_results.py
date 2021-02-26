"""Script for plotting results

Plots:
1. return vs independence per model
"""
from typing import Optional
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bdgym.scripts.exercise_assistant.utils as utils


CONSTANT_ATHLETE_POLICIES = ['random', 'random_weighted']


POLICY_NAME_MAP = {
    'discrete_donothing': "No Assistant",
    'random': 'Random',
    'random_weighted': 'Sampled\nAthlete',
    'weighted': 'Athlete',
    'PPO': 'PPO'
}


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


def plot_xy_err(ax, df, x_key, y_key, yerr_key, **kwargs):
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


def plot_metric_bar(ax, df, x_key, y_key, z_key, styles=None, yerr_key=None,
                    y_lims=None, x_labels=None, y_label=None, z_labels=None):
    print("\nplot_metric_bar, df:")
    print(df)
    print(df['athlete'])
    if y_label is None:
        y_label = y_key

    if x_labels is None:
        x_labels = df[x_key].unique().tolist()
    x_start = np.array(range(len(x_labels)))

    if z_labels is None:
        z_labels = df[z_key].unique()
    width = BAR_WIDTH_LIMIT / len(z_labels) - BAR_GAP

    bars = []
    yerr = None
    for i, z_label in enumerate(z_labels):
        print("\nPlotting:", z_label)
        z_df = df[df[z_key] == z_label]
        print(z_df)
        y = z_df[y_key]
        print("y:")
        print(y)
        x = x_start + (i * width)

        # Need to ensure results are in correct order
        zx_pos = []
        for zx in z_df[x_key].unique():
            zx_idx = x_labels.index(zx)
            zx_pos.append(x[zx_idx])
        x = zx_pos

        print("x:")
        print(x)
        if yerr_key is not None:
            yerr = [
                z_df[yerr_key],
                z_df[yerr_key]
            ]
        args = [x, y, width]
        kwargs = dict(
            yerr=yerr,
            align='edge',
            label=POLICY_NAME_MAP.get(z_label, z_label)
        )
        if styles is None:
            bar = ax.bar(*args, **kwargs)
        else:
            bar = ax.bar(*args, **kwargs, **next(styles))

        bars.append(bar)

    ax.set_ylabel(y_label, fontsize=FONTSIZE)
    ax.set_xticks(x_start + BAR_WIDTH_LIMIT / 2)
    x_labels = [
        POLICY_NAME_MAP.get(label, label) for label in df[x_key].unique()
    ]
    ax.set_xticklabels(x_labels, fontsize=FONTSIZE)
    if y_lims:
        ax.set_ylim(y_lims)

    return bars, [POLICY_NAME_MAP.get(label, label) for label in z_labels]


def plot_eval_return(ax, df):
    """Plot return for each athlete versus assistant """
    plot_kwargs = {
        "x_key": "athlete",
        "y_key": "return_mean",
        "yerr_key": "return_std",
        "z_key": "assistant",
        "y_label": "Mean Return +/- Std"
    }

    bars, labels = plot_metric_bar(ax, df, **plot_kwargs)
    return bars, labels


def plot_eval_deception(ax, df):
    """Plot return versus independence """
    plot_kwargs = {
        "x_key": "athlete",
        "y_key": "deception_mean",
        "yerr_key": "deception_std",
        "z_key": "assistant",
        "y_label": "Mean Deception +/- Std",
        "y_lims": (0.0, 1.0)
    }

    bars, labels = plot_metric_bar(ax, df, **plot_kwargs)
    return bars, labels


def main(args: Namespace):
    """Run plotting script """
    df = import_results(args.result_file)
    # Take average over seeds
    avg_df = df.groupby(
        ['assistant', 'athlete', 'independence']
    ).mean().reset_index()

    fig, axs = plt.subplots(1, 2)

    bars, labels = plot_eval_return(axs[0], avg_df)
    plot_eval_deception(axs[1], avg_df)

    fig.legend(
        bars,
        labels,
        loc='lower center',
        ncol=3,
        fontsize=FONTSIZE,
        framealpha=0.0,
    )

    fig.tight_layout(w_pad=3, h_pad=2)
    plt.subplots_adjust(bottom=0.2)
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("result_file", type=str,
                        help="Results file to plot")
    parser.add_argument("-fatp", "--fixed_athlete_policy",
                        type=str, default='weighted',
                        help="Athlete policy to plot (default='weighted')")
    parser.add_argument("-fasp", "--fixed_assistant_policy",
                        type=str, default='PPO',
                        help="Assistant policy to plot (default='PPO')")
    main(parser.parse_args())
