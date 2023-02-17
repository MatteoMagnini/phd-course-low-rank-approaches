import distutils.cmd
from setuptools import find_packages, setup
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from dataset import download_dataset, save_dataset, load_dataset
from figures import *
from figures.distributions import generate_distributions_plots
from statistic import *
from figures.distributions import PATH as FIGURES_DISTRIBUTIONS_PATH
from figures.data_visualisation import PATH as FIGURES_DATA_VISUALISATION_PATH
from dataset import PATH as DATA_PATH


class DownloadDataset(distutils.cmd.Command):
    """A custom command to download the dataset from the data folder."""

    description = 'loads datasets from the data folder'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print('Downloading dataset...')
        save_dataset(download_dataset())
        print('Dataset downloaded.')


class AnalyseDataset(distutils.cmd.Command):
    """A custom command to analyse the dataset from the data folder."""

    description = 'analyses datasets from the data folder'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print('Analyzing dataset...')
        dataset = center_dataset(load_dataset())
        print('Dataset loaded.')
        print('Number of rows:', len(dataset))
        print('Number of columns:', len(dataset.columns) - 1)
        print('Not bunkrupt companies:', len(dataset[dataset['Bankrupt?'] == 0]))
        print('Bunkrupt companies:', len(dataset[dataset['Bankrupt?'] == 1]))
        analise_std(dataset)
        print('Drawing correlation matrix...')
        correlation_matrix = dataset.corr()
        save_figure(generate_heatmap(correlation_matrix), 'correlation_matrix.svg')
        print('Drawing covariance matrix...')
        covariance_matrix = dataset.cov()
        save_figure(generate_heatmap(covariance_matrix), 'covariance_matrix.svg')
        print('Removing features that are not correlated with the label...')
        post_processed_dataset = remove_uncorrelated_features(dataset)
        print('Number of columns after removing uncorrelated features:', len(post_processed_dataset.columns) - 1)
        analise_std(post_processed_dataset)
        print('Drawing correlation matrix...')
        correlation_matrix = post_processed_dataset.corr()
        save_figure(generate_heatmap(correlation_matrix), 'correlation_matrix_post_processed.svg')
        print('Drawing covariance matrix...')
        covariance_matrix = post_processed_dataset.cov()
        save_figure(generate_heatmap(covariance_matrix), 'covariance_matrix_post_processed.svg')
        x = post_processed_dataset.iloc[:, :-1]
        y = post_processed_dataset.iloc[:, -1]
        print('Drawing distributions of variables...')
        distributions = generate_distributions_plots(x)
        for i, distribution in enumerate(distributions):
            save_figure(distribution, f'distributions_{i}.svg', FIGURES_DISTRIBUTIONS_PATH)
        most_correlated_features_with_labels = correlation_matrix.iloc[:, -1].abs().sort_values(ascending=False)
        top3_features = most_correlated_features_with_labels.index[1:4]
        save_figure(generate_3d_scatter_plot(x[top3_features], y), FIGURES_DATA_VISUALISATION_PATH / 'original_3d.svg')
        top2_features = most_correlated_features_with_labels.index[1:3]
        save_figure(generate_2d_scatter_plot(x[top2_features], y), FIGURES_DATA_VISUALISATION_PATH / 'original_2d.svg')
        print('Applying PCA...')
        pca = PCA(random_state=0)
        pca_dataset = pd.DataFrame(pca.fit_transform(x, y))
        pca_columns = [f'PC{i}' for i in range(1, len(pca_dataset.columns) + 1)]
        pca_dataset.columns = pca_columns
        pca_relative_var = pca_dataset.var() / pca_dataset.var().sum()
        corr = generate_features_components_correlation_matrix(x, pca_dataset)
        correlation_circles = generate_correlation_circles(corr, pca_relative_var)
        for i, circle in enumerate(correlation_circles):
            save_figure(circle, f'correlation_circle_{i}.svg')
        save_figure(generate_variance_histogram(pca_relative_var), 'pca_variance.svg')
        pca_dataset = pca_dataset.iloc[:, :6]
        print('Drawing 3D scatter plot for data visualisation...')
        for first_pc, second_pc in [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4),
                                    (2, 5), (3, 4), (3, 5), (4, 5)]:
            save_figure(generate_2d_scatter_plot(pca_dataset.iloc[:, [first_pc, second_pc]], y),
                        FIGURES_DATA_VISUALISATION_PATH / f'pc{first_pc+1}_pc{second_pc+1}.svg')
        print('Drawing 2D scatter plot for data visualisation...')
        for fist_pc, second_pc, third_pc in [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5), (0, 2, 3), (0, 2, 4), (0, 2, 5),
                                             (0, 3, 4), (0, 3, 5), (0, 4, 5), (1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 3, 4),
                                             (1, 3, 5), (1, 4, 5), (2, 3, 4), (2, 3, 5), (2, 4, 5), (3, 4, 5)]:
            save_figure(generate_3d_scatter_plot(pca_dataset.iloc[:, [fist_pc, second_pc, third_pc]], y),
                        FIGURES_DATA_VISUALISATION_PATH / f'pc{fist_pc+1}_pc{second_pc+1}_pc{third_pc+1}.svg')
        print('Saving PCA dataset...')
        pca_dataset.join(y).to_csv(DATA_PATH / 'pca_dataset.csv', index=False)
        print('Dataset analyzed.')


class Classification(distutils.cmd.Command):
    """A custom command to run the classification."""
    description = 'Run the classification'
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        print('Running classification...')

        def evaluate_classification(dataset):
            train, test = train_test_split(dataset, test_size=0.2, random_state=0, stratify=dataset.iloc[:, -1])
            train_x = train.iloc[:, :-1]
            train_y = train.iloc[:, -1]
            test_x = test.iloc[:, :-1]
            test_y = test.iloc[:, -1]
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(train_x, train_y)
            accuracy = model.score(test_x, test_y)
            y_pred = model.predict(test_x)
            f1 = f1_score(test_y, y_pred, average='weighted')
            precision = precision_score(test_y, y_pred, average='weighted')
            recall = recall_score(test_y, y_pred, average='weighted')
            print(f'Accuracy: {accuracy}')
            print(f'F1: {f1}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')

        print('Classification on the original dataset...')
        evaluate_classification(load_dataset())
        print('Classification on the PCA dataset...')
        evaluate_classification(pd.read_csv(DATA_PATH / 'pca_dataset.csv'))


setup(
    name='Low-Rank Approaches for Data Analysis',  # Required
    description='Project for the course of Low-Rank Approaches for Data Analysis',
    license='Apache 2.0 License',
    url='https://github.com/MatteoMagnini/phd-course-low-rank-approaches',
    author='Matteo Magnini',
    author_email='matteo.magnini@unibo.it',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='',  # Optional
    # package_dir={'': 'src'},  # Optional
    packages=find_packages(),  # Required
    include_package_data=True,
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.22.3',
        'scikit-learn>=1.0.2',
        'pandas>=1.4.2',
    ],  # Optional
    zip_safe=False,
    cmdclass={
        'download_dataset': DownloadDataset,
        'analyse_dataset': AnalyseDataset,
        'classification': Classification,
    },
)
