"""Script for plotting Exercise Assistant results """
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt

import bdgym.scripts.plot_utils as plot_utils


CONSTANT_ATHLETE_POLICIES = ['random', 'random_weighted']


POLICY_NAME_MAP = {
    'discrete_donothing': "No Assistant",
    'random': 'Random',
    'random_weighted': 'Sampled\nAthlete',
    'weighted': 'Athlete',
    'PPO': 'PPO'
}


def plot_eval_return(ax, df):
    """Plot return for each athlete versus assistant """
    plot_kwargs = {
        "x_key": "athlete",
        "y_key": "return_mean",
        "yerr_key": "return_std",
        "z_key": "assistant",
        "y_label": "Mean Return +/- Std",
        "label_map": POLICY_NAME_MAP
    }

    bars, labels = plot_utils.plot_metric_bar(ax, df, **plot_kwargs)
    return bars, labels


def plot_eval_overexertion(ax, df):
    """Plot return versus independence """
    plot_kwargs = {
        "x_key": "athlete",
        "y_key": "overexertion_prob",
        "z_key": "assistant",
        "y_label": "Overexertion Proportion",
        "y_lims": (0.0, 1.0),
        "label_map": POLICY_NAME_MAP
    }

    bars, labels = plot_utils.plot_metric_bar(ax, df, **plot_kwargs)
    return bars, labels


def plot_eval_deception(ax, df):
    """Plot return versus independence """
    plot_kwargs = {
        "x_key": "athlete",
        "y_key": "deception_mean",
        "yerr_key": "deception_std",
        "z_key": "assistant",
        "y_label": "Mean Deception +/- Std",
        "y_lims": (0.0, 1.0),
        "label_map": POLICY_NAME_MAP
    }

    bars, labels = plot_utils.plot_metric_bar(ax, df, **plot_kwargs)
    return bars, labels


def main(args: Namespace):
    """Run plotting script """
    df = plot_utils.import_results(args.result_file)
    # Take average over seeds
    avg_df = df.groupby(
        ['assistant', 'athlete', 'independence']
    ).mean().reset_index()

    fig, axs = plt.subplots(1, 3)

    bars, labels = plot_eval_return(axs[0], avg_df)
    plot_eval_overexertion(axs[1], avg_df)
    plot_eval_deception(axs[2], avg_df)

    fig.legend(
        bars,
        labels,
        loc='lower center',
        ncol=3,
        fontsize=plot_utils.FONTSIZE,
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
