from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression

PATH = Path(__file__).parents[0]


def compute_missing_values_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the frequency of missing values for each column.
    :param df: The dataset as a pandas DataFrame.
    :return: The frequency of missing values for each column.
    """
    return df.isnull().sum() / len(df)


def substitute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Substitute missing values following these criteria:
    - if the frequency of the missing values is less than 1%, then substitute with the mean of the column;
    - if the frequency of the missing values is greater than 1%, then substitute with the regression of the second most
    correlated column with the given column if the correlation is greater than 0.8 and less than 0.99 else substitute
    with the mean of the column (if the value of the correlated column is missing, then use the mean of the
    correlated column to perform the regression);
    - if the frequency of the missing values is greater than 10%, then drop the column.
    :param df: The dataset as a pandas DataFrame.
    :return: The dataset with missing values substituted.
    """
    missing_values_frequency = compute_missing_values_frequency(df)
    defensive_copy = df.copy()
    for column in df.columns:
        if missing_values_frequency[column] < 0.01:
            defensive_copy[column].fillna(df[column].mean(), inplace=True)
        elif missing_values_frequency[column] < 0.1:
            correlated_column, correlation = second_most_correlated_column(df, column)
            # The upperbound is 0.99 because the correlation of a column with a one that almost lineally dependent on it
            # is close to 1.0. This scenario is inconvenient because missing values appear in both columns at the same
            # time.
            if 0.8 < correlation < 0.99:
                all_correlated_values = df[correlated_column]
                regressor = get_trained_linear_regressor(all_correlated_values, df[column])
                missing_values_indices = df[column][df[column].isnull()].index
                correlated_values = all_correlated_values[missing_values_indices]
                # if correlated values contains nan values, then substitute with the mean of the column
                correlated_values.fillna(all_correlated_values.mean(), inplace=True)
                substitutions = regressor.predict(correlated_values.values.reshape(-1, 1))
                defensive_copy[column].fillna(pd.Series(substitutions), inplace=True)
            else:
                defensive_copy[column].fillna(df[column].mean(), inplace=True)
        else:
            defensive_copy.drop(column, axis=1, inplace=True)
    return defensive_copy


def normalise_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the dataset (min max scaling) except for the last column.
    :param df: The dataset as a pandas DataFrame.
    :return: The normalised dataset.
    """
    for column in df.columns[:-1]:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df


def second_most_correlated_column(df: pd.DataFrame, column: str) -> tuple[str, float]:
    """
    Compute the second most positive correlated column with the given column.
    :param df: The dataset as a pandas DataFrame.
    :param column: The column to compare.
    :return: The second most correlated column along with the correlation value.
    """
    second_most_correlated = df.corr().loc[column].sort_values(ascending=False)
    return second_most_correlated.index[1], second_most_correlated[1]


def get_trained_linear_regressor(train_x: pd.DataFrame, train_y: pd.DataFrame) -> LinearRegression:
    """
    Before the training the features and labels are preprocessed:
    - if there is a missing value in the features drop it and also drop the value in the labels at the same index;
    - if there is a missing value in the labels drop it and also drop the value in the features at the same index.
    :param train_x: The training features.
    :param train_y: The training labels.
    :return: The trained linear regressor.
    """
    missing_features_indices = train_x[train_x.isnull()].index
    missing_labels_indices = train_y[train_y.isnull()].index
    missing_indices = missing_features_indices.union(missing_labels_indices)
    train_x = pd.DataFrame(train_x.drop(missing_indices))
    train_y = train_y.drop(missing_indices)
    regressor = LinearRegression()
    regressor.fit(train_x, train_y)
    return regressor

