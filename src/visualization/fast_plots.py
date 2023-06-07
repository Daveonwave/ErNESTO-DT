import matplotlib.pyplot as plt
#import seaborn as sns


def plot_compared_data(dfs, variables, labels, x_axes, title, colors=None):
    """

    """

    if not colors:
        colors = ['violet', 'cyan', 'purple', 'magenta']

    plt.figure(figsize=(15,5))

    # Command for the grid
    plt.grid(visible=True, which='major', color='gray', alpha=0.25, linestyle='dashdot', lw=1.5)
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='beige', alpha=0.5, ls='-', lw=1)

    # Plot iteratively all the variables
    for i, df in enumerate(dfs):
        plt.plot(df[x_axes[i]], df[variables[i]], label=labels[i], color=colors[i])

    plt.title(title)
    plt.legend()
    plt.show()


def plot_separate_vars(df, variables, x_var, title, colors=None):
    """

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

        ax.plot(df[x_var], df[var], label=var, color=colors[i])
        ax.set_title(title + var)
        ax.legend()

    plt.show()