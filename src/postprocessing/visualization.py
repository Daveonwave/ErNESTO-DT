import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd

plot_types = ['scatter', 'line']


def ernesto_plotter(dfs: list,
                    variables: list,
                    labels: list,
                    x_axes: list,
                    x_labels: list,
                    y_labels: list,
                    markers: list,
                    line_styles: list,
                    plot_type: str = 'line',
                    title: str = '',
<<<<<<< HEAD
=======
                    sampling_rate: int = 1,
>>>>>>> online_learning
                    colors: list = None,
                    events: list = None,
                    save_fig: bool = False,
                    dest: Path = None, 
                    fig_name: str = '',
                    pic_format: str = 'png',
                    figsize: tuple =(15, 5),
                    tick_font_size: int = 16,
                    label_font_size: int = 18,
                    legend_font_size: int = 14
                    ):
    """
    Plot the data of multiple dataframes in the same figure for comparison.
    
    Args:
        dfs (list): list of dataframes to plot.
        variables (list): list of variables to plot.
        labels (list): list of labels for the legend.
        x_axes (list): list of x axes for the plot.
        x_labels (list): labels for the x axes.
        y_labels (list): labels for the y axes.
        markers (list): list of markers for the plot.
        line_styles (list): list of line styles for the plot.
        plot_type (str): type of plot (scatter or line).
        title (str): title of the plot.
<<<<<<< HEAD
=======
        sampling_rate (int): sampling rate of the data.
>>>>>>> online_learning
        colors (list, optional): list of colors for the plot. Defaults to None.
        events (list, optional): list of events to highlight in the plot. Defaults to None.
        save_fig (bool, optional): flag to save the figure or not. Defaults to False.
        dest (Path, optional): destination folder to save the figure. Defaults to None.
        fig_name (str, optional): name of the figure. Defaults to ''.
        pic_format (str, optional): format of the picture. Defaults to 'png'.
        figsize (tuple, optional): size of the figure. Defaults to (15, 5).
        tick_font_size (int, optional): font size of the ticks. Defaults to 16.
        label_font_size (int, optional): font size of the labels. Defaults to 18.
        legend_font_size (int, optional): font size of the legend. Defaults to 14.
    """
    if not colors:
        colors = ['violet', 'cyan', 'purple', 'magenta']
    
    assert plot_type in plot_types, "The plot type must be either 'scatter' or 'line'."

    fig, axes = plt.subplots(len(variables), 1, figsize=(figsize[0], figsize[1] * len(variables)))
    
    # Plot iteratively all the variables
    for i, var in enumerate(variables):

        # We need this assignement in the case of a single variable
        ax = axes if len(variables) == 1 else axes[i]

        # Command for the grid
        ax.grid(visible=True, which='major', color='gray', alpha=0.25, linestyle='dashdot', lw=1.5)
        ax.minorticks_on()
        ax.grid(visible=True, which='minor', color='beige', alpha=0.5, ls='-', lw=1)

        for j, df in enumerate(dfs):
            if plot_type == 'scatter':
<<<<<<< HEAD
                ax.scatter(df[x_axes[i]], df[var], label=labels[j], color=colors[j], s=0.1, rasterized=True)
            else:
                ax.plot(df[x_axes[i]], df[var], label=labels[j], color=colors[j], marker=markers[j], markevery=3000, linestyle=line_styles[j])
=======
                ax.scatter(df[x_axes[i]][::sampling_rate], df[var][::sampling_rate], label=labels[j], color=colors[j], s=0.1, rasterized=True)
            else:
                ax.plot(df[x_axes[i]][::sampling_rate], df[var][::sampling_rate], label=labels[j], color=colors[j], marker=markers[j], markevery=3000, linestyle=line_styles[j])
>>>>>>> online_learning
        
        ax.tick_params(labelsize=tick_font_size)
        ax.set_xlabel(x_labels[i], size=label_font_size)
        ax.set_ylabel(y_labels[i], size=label_font_size)
        
        if plot_type == 'scatter':
            ax.legend(markerscale=5, scatterpoints=5, fontsize=legend_font_size)
        else:
            ax.legend(fontsize=legend_font_size)

        if events:
            for j, event in enumerate(events):
                facecolor = 'lightgray' if j % 2 == 0 else 'gray'
                ax.axvspan(event[0], event[1], facecolor=facecolor, alpha=0.3)
                ax.axvline(x=event[1], color='black', alpha=0.2)
    
        if save_fig:
            Path(dest / 'plots').mkdir(parents=True, exist_ok=True) 
            
            if pic_format == 'pgf':
                matplotlib.rcParams.update({
                    "pgf.texsystem": "pdflatex",
                    'font.family': 'serif',
                    'text.usetex': True,
                    'pgf.rcfonts': False,
                })
            
            plot_name = fig_name + '_{}.{}'.format(var, pic_format)
            file_path = dest / 'plots' / plot_name
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(file_path, format=pic_format, transparent=True, dpi=200, bbox_inches=extent.expanded(2, 2))

    plt.show()
    

def plot_separate_vars(df: pd.DataFrame,
                       variables: list,
                       labels: list,
                       x_ax: list,
                       x_label: str,
                       y_labels: str,
<<<<<<< HEAD
=======
                       sampling_rate: int = 1,
>>>>>>> online_learning
                       colors: list = None,
                       events: list = None,
                       save_fig: bool = False,
                       dest: Path = None, 
                       fig_name: str = '',
                       pic_format: str = 'png',
                       tick_font_size: int = 16,
                       label_font_size: int = 18,
                       legend_font_size: int = 14
                       ):
    """
    Plot the data of multiple variables in separate subplots.
    
    Args:
        df (pd.DataFrame): dataframe containing the data.
        variables (list): list of variables to plot.
        labels (list): list of labels for the legend.
        x_ax (list): list of x axes for the plot.
        x_label (str): x label.
        y_labels (str): y label of each subplot.
<<<<<<< HEAD
=======
        sampling_rate (int): sampling rate of the data.
>>>>>>> online_learning
        titles (str): title of each subplot.
        colors (list, optional): list of colors for the plot. Defaults to None.
        events (list, optional): list of events to highlight in the plot. Defaults to None.
        save_fig (bool, optional): flag to save the figure or not. Defaults to False.
        dest (Path, optional): destination folder to save the figure. Defaults to None.
        fig_name (str, optional): name of the figure. Defaults to ''.
        pic_format (str, optional): format of the picture. Defaults to 'png'.
    """
    if not colors:
        colors = ['cyan', 'violet', 'purple', 'magenta']

    fig, axes = plt.subplots(len(variables), 1, figsize=(15, 4 * len(variables)), sharex=True)

    # Plot iteratively all the variables
    for i, var in enumerate(variables):

        # We need this assignement in the case of a single variable
        ax = axes if len(variables) == 1 else axes[i]

        # Command for the grid
        ax.grid(visible=True, which='major', color='gray', alpha=0.25, linestyle='dashdot', lw=1.5)
        ax.minorticks_on()
        ax.grid(visible=True, which='minor', color='beige', alpha=0.5, ls='-', lw=1)

<<<<<<< HEAD
        ax.plot(df[x_ax], df[var], label=labels[i], color=colors[i])
=======
        ax.plot(df[x_ax][::sampling_rate], df[var][::sampling_rate], label=labels[i], color=colors[i])
>>>>>>> online_learning
        
        ax.tick_params(labelsize=tick_font_size)
        ax.set_xlabel(x_label, size=label_font_size)
        ax.set_ylabel(y_labels[i], size=label_font_size)
        ax.legend(fontsize=legend_font_size)

        if events:
            for j, event in enumerate(events):
                facecolor = 'lightgray' if j % 2 == 0 else 'gray'
                ax.axvspan(event[0], event[1], facecolor=facecolor, alpha=0.3)
                ax.axvline(x=event[1], color='black', alpha=0.2)

    plt.xlabel(x_label)
    plt.ticklabel_format(style='plain')
    
    if save_fig:
        if pic_format == 'pgf':
            matplotlib.rcParams.update({
                "pgf.texsystem": "pdflatex",
                'font.family': 'serif',
                'text.usetex': True,
                'pgf.rcfonts': False,
            })
        
        file_path = dest / fig_name
        plt.savefig(file_path, format=pic_format, transparent=True, dpi=200)

    plt.show()
