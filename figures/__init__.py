import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

PATH = Path(__file__).parents[0]


def generate_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Generate a heatmap of the correlation or covariance matrix.
    :param df: The dataset as a pandas DataFrame.
    :return: The heatmap as a matplotlib Figure.
    """
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(20, 20))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, square=True, cbar_kws={"shrink": .5})
    return f


def generate_distributions_plots(df: pd.DataFrame) -> list[plt.Figure]:
    """
    Generate a list of different plots, each organised into subplots (a matrix of 4 x 4 subplots).
    A subplot is a distribution plot of a feature (for 64 features there are 4 plots of 16 subplots each).
    If the number of features is not a multiple of 16, the remaining subplots are empty.
    :param df: The dataset as a pandas DataFrame.
    :return: The distribution plots as a list of matplotlib Figure.
    """
    plots = []
    for i in range(0, len(df.columns), 16):
        f, axes = plt.subplots(4, 4, figsize=(20, 20))
        for j in range(0, 16):
            if i + j < len(df.columns):
                sns.distplot(df.iloc[:, i + j], ax=axes[j // 4, j % 4])
        plots.append(f)
    return plots


def generate_variance_histogram(variances: np.ndarray) -> plt.Figure:
    """
    Generate a histogram of the variances of the principal components.
    The plot does not show the variance of principal components whose variance is less than 0.01.
    The plot includes also a line that represents the sum of all previous variances.
    :param variances: The variances of the principal components.
    :return: The histogram as a matplotlib Figure.
    """
    f, ax = plt.subplots(figsize=(20, 10))
    variances = variances[variances > 0.01]
    ax.bar(range(1, len(variances) + 1), variances.cumsum(), color='red')
    ax.bar(range(1, len(variances) + 1), variances)
    x_ticks = [i for i in range(1, len(variances) + 1)]
    ax.set_xticks(x_ticks)
    return f


def generate_3d_scatter_plot(df: pd.DataFrame, labels: np.ndarray) -> plt.Figure:
    """
    Generate a 3D scatter plot of the first three principal components.
    Colors are assigned to the points based on the labels (0 is ble and 1 is red).
    :param df: The dataset as a pandas DataFrame.
    :param labels: The labels of the dataset.
    :return: The 3D scatter plot as a matplotlib Figure.
    """
    f = plt.figure(figsize=(20, 10))
    ax = f.add_subplot(111, projection='3d')
    ax.scatter(df.iloc[:, 2], df.iloc[:, 1], df.iloc[:, 0], c=labels, cmap='bwr')
    return f


def generate_2d_scatter_plot(df: pd.DataFrame, labels: np.ndarray) -> plt.Figure:
    """
    Generate a 2D scatter plot of the first two principal components.
    Colors are assigned to the points based on the labels (0 is ble and 1 is red).
    :param df: The dataset as a pandas DataFrame.
    :param labels: The labels of the dataset.
    :return: The 2D scatter plot as a matplotlib Figure.
    """
    f, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='bwr')
    return f


def generate_correlation_circle_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Generate a correlation circle plot of the first two principal components.
    :param df: The dataset as a pandas DataFrame.
    :return: The correlation circle plot as a matplotlib Figure.
    """
    f, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Correlation Circle Plot')
    for i in range(len(df)):
        ax.arrow(0, 0, df.iloc[i, 0], df.iloc[i, 1], color='red', alpha=0.5)
        if math.sqrt(df.iloc[i, 0]**2 + df.iloc[i, 1]**2) > 0.25:
            ax.text(df.iloc[i, 0], df.iloc[i, 1], df.index[i], fontsize=10)
    ax.add_patch(plt.Circle((0, 0), 1, color='black', fill=False))
    return f


def save_figure(plot: plt.Figure, filename: str) -> None:
    """
    Save a plot to a file.
    :param plot: The plot to save.
    :param filename: The name of the file.
    :return: None
    """
    plot.savefig(PATH / filename, format='svg')
    plot.clf()
