import distutils.cmd
import pandas as pd
from setuptools import find_packages, setup
from sklearn.decomposition import PCA
from dataset import download_dataset, save_dataset, load_dataset
from figures import save_figure, generate_heatmap, generate_variance_histogram, \
    generate_3d_scatter_plot, generate_correlation_circle_plot, generate_2d_scatter_plot, \
    generate_correlation_sphere_plot, generate_correlation_circles
from figures.distributions import generate_distributions_plots
from statistic import compute_missing_values_frequency, normalise_dataset, substitute_missing_values, \
    standardise_dataset, generate_features_components_correlation_matrix, remove_uncorrelated_features, analise_std
from figures.distributions import PATH as FIGURES_DISTRIBUTIONS_PATH


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
        dataset = load_dataset()
        print('Dataset loaded.')
        print('Number of rows:', len(dataset))
        print('Number of columns:', len(dataset.columns) - 1)
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
        save_figure(generate_3d_scatter_plot(x[top3_features], y), 'original_3d_data_representation.svg')
        top2_features = most_correlated_features_with_labels.index[1:3]
        save_figure(generate_2d_scatter_plot(x[top2_features], y), 'original_2d_data_representation.svg')
        print('Applying PCA...')
        pca = PCA(random_state=0)
        pca_dataset = pd.DataFrame(pca.fit_transform(x, y))
        pca_relative_std = pca_dataset.std() / pca_dataset.std().sum()
        corr = generate_features_components_correlation_matrix(x, pca_dataset)
        correlation_circles = generate_correlation_circles(corr, pca_relative_std)
        for i, circle in enumerate(correlation_circles):
            save_figure(circle, f'correlation_circle_{i}.svg')
        save_figure(generate_variance_histogram(pca_relative_std), 'pca_variance.svg')
        pca_dataset = pca_dataset.iloc[:, :6]
        most_correlated_pca_components_with_labels = pca_dataset.join(y).corr().abs().iloc[:, -1].sort_values(ascending=False)
        top3_pca = most_correlated_pca_components_with_labels.index[1:4]
        save_figure(generate_3d_scatter_plot(pca_dataset[top3_pca], y), 'pca_3d_scatter_plot.svg')
        top2_pca = most_correlated_pca_components_with_labels.index[1:3]
        save_figure(generate_2d_scatter_plot(pca_dataset[top2_pca], y), 'pca_2d_scatter_plot.svg')
        print('Dataset analyzed.')


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
    },
)
