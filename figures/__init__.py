from __future__ import annotations
import math
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

PATH = Path(__file__).parents[0]

CUMULATIVE_VARIANCE_THRESHOLD = 0.9
PC_FEATURE_THRESHOLD = 0.3
IGNORE_VARIANCE_THRESHOLD = 1E-3


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


def generate_variance_histogram(variances: np.ndarray) -> plt.Figure:
    """
    Generate a histogram of the variances of the principal components.
    The plot does not show the variance of principal components whose variance is less than IGNORE_VARIANCE_THRESHOLD.
    The plot includes also a line that represents the sum of all previous variances.
    :param variances: The variances of the principal components.
    :return: The histogram as a matplotlib Figure.
    """
    f, ax = plt.subplots(figsize=(20, 10))
    variances = variances[variances > IGNORE_VARIANCE_THRESHOLD]
    ax.bar(range(1, len(variances) + 1), variances.cumsum(), color='red')
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', ha='center')
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
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=labels, cmap='bwr')
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_zlabel(df.columns[2])
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
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    return f


def generate_correlation_circle_plot(df: pd.DataFrame, first_pc='PC1', second_pc='PC2') -> plt.Figure:
    """
    Generate a correlation circle plot of the first two principal components.
    :param df: The dataset as a pandas DataFrame.
    :param first_pc: The name of the first principal component.
    :param second_pc: The name of the second principal component.
    :return: The correlation circle plot as a matplotlib Figure.
    """
    f, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_title('Correlation Circle Plot')
    for i in range(len(df)):
        if abs(df.iloc[i, 0]) > PC_FEATURE_THRESHOLD or abs(df.iloc[i, 1]) > PC_FEATURE_THRESHOLD:
            ax.arrow(0, 0, df.iloc[i, 0], df.iloc[i, 1], color='red', alpha=0.8, width=0.005)
            ax.text(df.iloc[i, 0], df.iloc[i, 1], df.index[i], fontsize=10)
    ax.add_patch(plt.Circle((0, 0), 1, color='black', fill=False))
    ax.axvline(0, color='black')
    ax.axhline(0, color='black')
    ax.set_xlabel(first_pc)
    ax.set_ylabel(second_pc)
    return f


def generate_correlation_circles(df: pd.DataFrame, variances: np.ndarray) -> list[plt.Figure]:
    """
    Generate a set figures representing correlation circles.
    Each circle is a pair of principal components.
    Principal components are ordered by their variance.
    Ignore principal components if the cumulative variance is greater than CUMULATIVE_VARIANCE_THRESHOLD
    :param df: The dataset as a pandas DataFrame.
    :param variances: The variances of the principal components.
    :return: The correlation circles plot as a matplotlib Figure.
    """
    cumulative_variance = 0
    plots = []
    for i, j in zip(range(0, len(variances), 2), range(1, len(variances), 2)):
        if cumulative_variance > CUMULATIVE_VARIANCE_THRESHOLD:
            break
        cumulative_variance += variances[i] + variances[j]
        plots.append(generate_correlation_circle_plot(df.iloc[:, [i, j]], f'PC{i + 1}', f'PC{j + 1}'))
    return plots


def generate_correlation_sphere_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Generate a correlation sphere plot of the first three principal components.
    :param df: The dataset as a pandas DataFrame.
    :return: The correlation sphere plot as a matplotlib Figure.
    """
    f = plt.figure(figsize=(15, 15))
    ax = f.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Correlation Sphere Plot')
    for i in range(len(df)):
        ax.plot([0, df.iloc[i, 0]], [0, df.iloc[i, 1]], [0, df.iloc[i, 2]], color='red', alpha=0.5)
        if math.sqrt(df.iloc[i, 0] ** 2 + df.iloc[i, 1] ** 2 + df.iloc[i, 2] ** 2) > 0.25:
            ax.text(df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.index[i], fontsize=10)

    def generate_sphere(radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a sphere.
        :param radius: The radius of the sphere.
        :return: The coordinates of the sphere.
        """
        u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
        x = radius * np.cos(u) * np.sin(v)
        y = radius * np.sin(u) * np.sin(v)
        z = radius * np.cos(v)
        return x, y, z

    ax.plot_surface(*generate_sphere(1), color='black', alpha=0.1)
    return f


def save_figure(plot: plt.Figure, filename: str, path=PATH) -> None:
    """
    Save a plot to a file.
    :param plot: The plot to save.
    :param filename: The name of the file.
    :param path: The path to the file.
    :return: None
    """
    if os.path.exists(path / filename):
        os.remove(path / filename)
    plot.savefig(path / filename, format='svg')
