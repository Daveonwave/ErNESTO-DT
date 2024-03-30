import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd

pic_format = 'png'  # pdf / svg


def plot_compared_data(dest: Path,
                       dfs: list,
                       variables: list,
                       labels: list,
                       x_axes: list,
                       title: str,
                       colors: list = None
                       ):
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
        plt.scatter(df[x_axes[i]], df[variables[i]], label=labels[i], color=colors[i], s=0.1)
        # plt.plot(df[x_axes[i]], df[variables[i]], label=labels[i], color=colors[i])

    plt.title(title)
    plt.legend()

    file_path = dest / title
    plt.savefig(file_path)


def plot_separate_vars(dest: Path,
                       df: pd.DataFrame,
                       variables: list,
                       x_ax: list,
                       title: str,
                       colors: list = None,
                       events: list = None
                       ):
    """

    Args:
        dest (Path):
        df (pd.DataFrame):
        variables (list):
        x_ax (list):
        title (str):
        colors (list):
        events (list)
    """
    if not colors:
        colors = ['cyan', 'violet', 'purple', 'magenta']

    fig, axes = plt.subplots(len(variables), 1, figsize=(15, 3.5 * len(variables)), sharex=True)

    # Plot iteratively all the variables
    for i, var in enumerate(variables):

        # We need this assignement in the case of a single variable
        ax = axes if len(variables) == 1 else axes[i]

        # Command for the grid
        ax.grid(visible=True, which='major', color='gray', alpha=0.25, linestyle='dashdot', lw=1.5)
        ax.minorticks_on()
        ax.grid(visible=True, which='minor', color='beige', alpha=0.5, ls='-', lw=1)

        ax.plot(df[x_ax], df[var], label=var, color=colors[i])
        ax.set_title(title + ': ' + var)
        ax.legend()

        if events:
            for j, event in enumerate(events):
                facecolor = 'lightgray' if j % 2 == 0 else 'gray'
                ax.axvspan(event[0], event[1], facecolor=facecolor, alpha=0.3)
                ax.axvline(x=event[1], color='black', alpha=0.2)

    file_path = dest / title
    fig.savefig(file_path)
