"""Script for plotting Exercise Assistant results """
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt

import bdgym.scripts.plot_utils as plot_utils


CONSTANT_DRIVER_POLICIES = ['random', 'standard', 'aggressive']


POLICY_NAME_MAP = {
    'random': 'Random',
    'changing': 'Sampled\nDriver',
    'standard': 'Standard\nDriver',
    'aggressive': 'Aggressive\nDriver',
    'PPO': 'PPO',
    "None": "No Assistant",
}


DECEPTION_FEATURES = ['x', 'y', 'vx', 'vy']


def plot_eval_return(ax, df):
    """Plot return for each athlete versus assistant """
    plot_kwargs = {
        "x_key": "driver_label",
        "y_key": "return_mean",
        "yerr_key": "return_std",
        "z_key": "assistant",
        "y_label": "Mean Return +/- Std",
        "label_map": POLICY_NAME_MAP,
        "z_labels": ["PPO", "None"]
    }

    bars, labels = plot_utils.plot_metric_bar(ax, df, **plot_kwargs)
    return bars, labels


def plot_eval_collisions(ax, df):
    """Plot return for each athlete versus assistant """
    plot_kwargs = {
        "x_key": "driver_label",
        "y_key": "collision_prob",
        "z_key": "assistant",
        "y_label": "Collision Proportion",
        "label_map": POLICY_NAME_MAP,
        "z_labels": ["PPO", "None"]
    }

    bars, labels = plot_utils.plot_metric_bar(ax, df, **plot_kwargs)
    return bars, labels


def plot_eval_deception_feature(ax, df, feature):
    """Plot return versus independence """
    plot_kwargs = {
        "x_key": "driver_label",
        "y_key": f"deception_mean_{feature}",
        "yerr_key": f"deception_std_{feature}",
        "z_key": "assistant",
        "y_label": "Mean Deception +/- Std",
        "y_lims": (0.0, 1.0),
        "label_map": POLICY_NAME_MAP
    }

    bars, labels = plot_utils.plot_metric_bar(ax, df, **plot_kwargs)
    ax.set_title(f"Feature = {feature}")
    return bars, labels


def driver_independence_label(row):
    """Returns combined driver-independence label given DF row """
    driver = row['driver']
    independence = row['independence']
    if driver in ['random', 'changing']:
        return driver
    return f"{driver}\n{independence:.2f}"


def main(args: Namespace):
    """Run plotting script """
    df = plot_utils.import_results(args.result_file)

    # Combine driver and independence into a single column
    df['driver_label'] = df.apply(
        lambda row: driver_independence_label(row), axis=1
    )

    # Take average over seeds
    avg_df = df.groupby(
        ['assistant', 'driver_label']
    ).mean().reset_index()

    # fig, axs = plt.subplots(2, len(DECEPTION_FEATURES))
    fig, axs = plt.subplots(1, 2)

    bars, labels = plot_eval_return(axs[0], avg_df)
    plot_eval_collisions(axs[1], avg_df)

    fig.legend(
        bars,
        labels,
        loc='lower center',
        ncol=3,
        fontsize=plot_utils.FONTSIZE,
        framealpha=0.0,
    )

    # fig.tight_layout(w_pad=3, h_pad=2)
    fig.tight_layout()

    dec_df = df[df['assistant'] != 'None']

    dec_fig, dec_axs = plt.subplots(1, len(DECEPTION_FEATURES), sharey=True)
    for i, feature in enumerate(DECEPTION_FEATURES):
        plot_eval_deception_feature(dec_axs[i], dec_df, feature)

    for ax in dec_axs[1:]:
        ax.set_ylabel(None)

    dec_fig.tight_layout()

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
