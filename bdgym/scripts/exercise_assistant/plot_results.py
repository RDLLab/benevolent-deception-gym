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


def import_results(results_file: str,
                   sep: Optional[str] = None) -> pd.DataFrame:
    """Import results from file into Pandas DataFrame """
    if sep is None:
        sep = "\t"
    return pd.read_csv(results_file, sep=sep)


def plot_return(ax, df):
    """Plot return versus independence """
    athlete_policies = df['athlete'].unique()
    assistant_policies = df['assistant'].unique()

    for athlete in athlete_policies:
        athlete_df = df[df['athlete'] == athlete]
        for assistant in assistant_policies:
            assistant_df = athlete_df[athlete_df['assistant'] == assistant]
            if len(assistant_df) == 0:
                continue

            assistant_df = assistant_df.groupby(['independence'])
            x = assistant_df['independence']
            y = assistant_df['return_mean'].mean()
            y_err = assistant_df['return_std'].mean()

            print(assistant_df.mean())

            if athlete in ['random', 'random_weighted']:
                x = np.linspace(0.0, 1.0, 11)
                y = np.full(11, y.to_numpy())
                y_err = np.full(11, y_err.to_numpy())

            ax.plot(x, y, label=f"{assistant}-{athlete}")
            ax.fill_between(x, y-y_err, y+y_err, alpha=0.5)

    ax.set_xlabel('Athlete Independence')
    ax.set_ylabel("Mean Return +/- Std")
    ax.legend()


def plot_deception(ax, df):
    """Plot return versus independence """
    athlete_policies = df['athlete'].unique()
    assistant_policies = df['assistant'].unique()

    for athlete in athlete_policies:
        athlete_df = df[df['athlete'] == athlete]
        for assistant in assistant_policies:
            assistant_df = athlete_df[athlete_df['assistant'] == assistant]
            if len(assistant_df) == 0:
                continue

            assistant_df = assistant_df.groupby(['independence'])
            x = assistant_df['independence']
            y = assistant_df['deception_mean'].mean()
            y_err = assistant_df['deception_std'].mean()

            if athlete in ['random', 'random_weighted']:
                x = np.linspace(0.0, 1.0, 11)
                y = np.full(11, y.to_numpy())
                y_err = np.full(11, y_err.to_numpy())

            ax.plot(x, y, label=f"{assistant}-{athlete}")
            ax.fill_between(x, y-y_err, y+y_err, alpha=0.5)

    ax.set_xlabel('Athlete Independence')
    ax.set_ylabel("Mean Deception +/- Std")
    ax.set_ylim(0.0, 1.0)
    ax.legend()


def main(args: Namespace):
    """Run plotting script """
    df = import_results(args.result_file)

    fig, axs = plt.subplots(1, 2)

    plot_return(axs[0], df)
    plot_deception(axs[1], df)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--result_file", type=str,
                        help="Results file to plot")
    parser.add_argument("-fatp", "--fixed_athlete_policy",
                        type=str, default='weighted',
                        help="Athlete policy to plot (default='weighted')")
    parser.add_argument("-fasp", "--fixed_assistant_policy",
                        type=str, default='PPO',
                        help="Assistant policy to plot (default='PPO')")
    main(parser.parse_args())
