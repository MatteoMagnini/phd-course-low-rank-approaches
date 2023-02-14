import math
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression

PATH = Path(__file__).parents[0]

CORR_THRESHOLD = 1E-2


def compute_missing_values_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the frequency of missing values for each column.
    :param df: The dataset as a pandas DataFrame.
    :return: The frequency of missing values for each column.
    """
    return df.isnull().sum() / len(df)


def analise_std(df: pd.DataFrame) -> None:
    """
    Analyse the standard deviation of each column.
    :param df: The dataset as a pandas DataFrame.
    :return: None
    """
    df_features = df.iloc[:, :-1]
    standard_deviations = df_features.std()
    percentage_std = standard_deviations / standard_deviations.sum() * 100
    one_percent = len(percentage_std[percentage_std < 1])
    print(f'\nFeatures with standard deviation less than 1% ({one_percent})')
    if one_percent < 10:
        for idx, column in enumerate(df_features.columns):
            if percentage_std[idx] < 1:
                print(f'{column} - standard deviation: {standard_deviations[idx]} - percentage: {percentage_std[idx]}')
    one_ten_percent = len(percentage_std[(1 <= percentage_std) & (percentage_std < 10)])
    print(f'\nFeatures with standard deviation greater than 1% and less than 10% ({one_ten_percent})')
    if one_ten_percent < 10:
        for idx, column in enumerate(df_features.columns):
            if 1 <= percentage_std[idx] < 10:
                print(f'{column} - standard deviation: {standard_deviations[idx]} - percentage: {percentage_std[idx]}')
    ten_percent = len(percentage_std[percentage_std >= 10])
    print(f'\nFeatures with standard deviation greater than 10% ({ten_percent}):')
    if ten_percent < 10:
        for idx, column in enumerate(df_features.columns):
            if percentage_std[idx] >= 10:
                print(f'{column} - standard deviation: {standard_deviations[idx]} - percentage: {percentage_std[idx]}')
    print('\n')


# Not used
def substitute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Substitute missing values following these criteria:
    - if the frequency of the missing values is less than 1%, then substitute with the mean of the column;
    - if the frequency of the missing values is greater than 1%, then substitute with the regression of the second most
      correlated column with the given column:
      - if the correlation is greater than 0.8 and less than 0.99 else substitute with the mean of the column;
      - if the value of the correlated column is missing use the mean of the correlated column instead;
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
                substitutions = {k: v for k, v in zip(missing_values_indices, substitutions)}
                defensive_copy[column].fillna(pd.Series(substitutions), inplace=True)
            else:
                defensive_copy[column].fillna(df[column].mean(), inplace=True)
        else:
            defensive_copy.drop(column, axis=1, inplace=True)
    return defensive_copy


def remove_uncorrelated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the features that are not correlated with the label with threshold t1 and that are not correlated with other
     features with threshold t2.
    :param df: The dataset as a pandas DataFrame.
    :return: The dataset with the uncorrelated features removed.
    """
    correlation = df.corr().iloc[-1]
    return df[[c for c in df.columns if (abs(correlation[c]) > CORR_THRESHOLD) and not math.isnan(correlation[c])]]


def normalise_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the dataset (min max scaling) except for the last column.
    :param df: The dataset as a pandas DataFrame.
    :return: The normalised dataset.
    """
    for column in df.columns[:-1]:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df


def standardise_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise the dataset except for the last column.
    :param df: The dataset as a pandas DataFrame.
    :return: The standardised dataset.
    """
    for column in df.columns[:-1]:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
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


def generate_features_components_correlation_matrix(df_features: pd.DataFrame, df_pca: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a correlation matrix between the features and the components of the PCA.
    :param df_features: The dataset with the features.
    :param df_pca: The dataset with the components of the PCA.
    :return: The correlation matrix.
    """
    result = pd.DataFrame()
    for i, feature in enumerate(df_features.columns):
        for j, component in enumerate(df_pca.columns):
            result.loc[feature, component] = df_features[feature].corr(df_pca[component])
    result.index = df_features.columns
    result.columns = ['PC_' + str(i+1) for i in range(len(df_pca.columns))]
    return result
