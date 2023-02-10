from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd

PATH = Path(__file__).parents[0]


"""
Because currently the Taiwanese Bankruptcy Prediction dataset is not available on the UCI repository,
I rely on the Polish Companies Bankruptcy dataset due to its similarity in domain.
"""
UCI_DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases'
DATA_URL = UCI_DATA_URL + '/00365/data.zip'
FILE_NAMES = ['1year.arff', '2year.arff', '3year.arff', '4year.arff', '5year.arff']
SEPARATOR = ','
DATASET_FILE_NAME = 'data.csv'
SKIP = 69


def download_dataset() -> pd.DataFrame:
    """
    Download the dataset from the UCI repository.
    :return: The dataset as a pandas DataFrame.
    """
    datasets = []
    with urlopen(DATA_URL) as response:
        with ZipFile(BytesIO(response.read())) as zip_file:
            for file_name in FILE_NAMES:
                with zip_file.open(file_name) as arff_file:
                    df = pd.read_csv(arff_file, sep=SEPARATOR, skiprows=SKIP, header=None, na_values='?')
                    datasets.append(df)
    return pd.concat(datasets)


def save_dataset(df: pd.DataFrame) -> None:
    """
    Save the dataset to a CSV file.
    :param df: The dataset as a pandas DataFrame.
    :return: None
    """
    df.to_csv(PATH / DATASET_FILE_NAME, index=False)


def load_dataset() -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    Because the first column represents the label, shift it to the last position.
    :return: The dataset as a pandas DataFrame.
    """
    df = pd.read_csv(PATH / DATASET_FILE_NAME)
    label = df.iloc[:, 0]
    features = df.iloc[:, 1:]
    return features.join(label)
