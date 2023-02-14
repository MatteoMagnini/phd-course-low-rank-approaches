from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

PATH = Path(__file__).parents[0]


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
                sns.histplot(df.iloc[:, i + j], ax=axes[j // 4, j % 4], bins="sqrt")
        plots.append(f)
    return plots
