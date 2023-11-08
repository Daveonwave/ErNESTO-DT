import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd


#import seaborn as sns


def plot_compared_data(dest: Path,
                       dfs: list,
                       variables: list,
                       labels: list,
                       x_axes: list,
                       title: str,
                       colors: list = None):
    """

    Args:
        dest (Path): destination folder (usually within output folder of current experiment)
        dfs (list): simulated and ground data
        variables (list):
        labels (list):
        x_axes (list):
        title (str):
        colors (list):
    """
    if not colors:
        colors = ['violet', 'cyan', 'purple', 'magenta']

    plt.figure(figsize=(15, 5))

    # Command for the grid
    plt.grid(visible=True, which='major', color='gray', alpha=0.25, linestyle='dashdot', lw=1.5)
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='beige', alpha=0.5, ls='-', lw=1)

    # Plot iteratively all the variables
    for i, df in enumerate(dfs):
        plt.plot(df[x_axes[i]], df[variables[i]], label=labels[i], color=colors[i])

    plt.title(title)
    plt.legend()

    file_path = dest / title
    plt.savefig(file_path)


def plot_separate_vars(dest: Path,
                       df: pd.DataFrame,
                       variables: list,
                       x_ax: list,
                       title: str,
                       colors: list = None):
    """

    Args:
        dest (Path):
        df (pd.DataFrame):
        variables (list):
        x_ax (list):
        title (str):
        colors (list):
    """
    if not colors:
        colors = ['cyan', 'violet', 'purple', 'magenta']

    fig, axes = plt.subplots(len(variables), 1, figsize=(15, 3.5 * len(variables)), sharex=True)

    # Plot iteratively all the variables
    for i, var in enumerate(variables):

        # We need this assignement in the case of a single variable
        if len(variables) == 1:
            ax = axes
        else:
            ax = axes[i]

        # Command for the grid
        ax.grid(b=True, which='major', color='gray', alpha=0.25, linestyle='dashdot', lw=1.5)
        ax.minorticks_on()
        ax.grid(b=True, which='minor', color='beige', alpha=0.5, ls='-', lw=1)

        ax.plot(df[x_ax], df[var], label=var, color=colors[i])
        ax.set_title(title + var)
        ax.legend()

    file_path = dest / title
    plt.savefig(file_path)
