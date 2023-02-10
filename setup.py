import distutils.cmd
import pandas as pd
from setuptools import find_packages, setup
from sklearn.decomposition import PCA
from dataset import download_dataset, save_dataset, load_dataset
from figures import save_figure, generate_heatmap, generate_distributions_plots, generate_variance_histogram, \
    generate_3d_scatter_plot, generate_correlation_circle_plot, generate_2d_scatter_plot
from statistic import compute_missing_values_frequency, normalise_dataset, substitute_missing_values, \
    standardise_dataset, generate_features_components_correlation_matrix, remove_uncorrelated_features


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
        correlation_matrix = dataset.corr()
        save_figure(generate_heatmap(correlation_matrix), 'correlation_matrix.svg')
        covariance_matrix = dataset.cov()
        save_figure(generate_heatmap(covariance_matrix), 'covariance_matrix.svg')

        # distribution_plots = generate_distributions_plots(dataset)
        # for i, plot in enumerate(distribution_plots):
        #     save_figure(plot, f'distribution_plot_{i+1}.png')

        post_processed_dataset = remove_uncorrelated_features(dataset)
        correlation_matrix = post_processed_dataset.corr()
        save_figure(generate_heatmap(correlation_matrix), 'correlation_matrix_post_processed.svg')
        covariance_matrix = post_processed_dataset.cov()
        save_figure(generate_heatmap(covariance_matrix), 'covariance_matrix_post_processed.svg')

        x = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        most_correlated_features_with_labels = correlation_matrix.iloc[:, -1].abs().sort_values(ascending=False)
        top3_features = most_correlated_features_with_labels.index[1:4]
        save_figure(generate_3d_scatter_plot(x[top3_features], y), 'original_3d_data_representation.svg')
        top2_features = most_correlated_features_with_labels.index[1:3]
        save_figure(generate_2d_scatter_plot(x[top2_features], y), 'original_2d_data_representation.svg')

        pca = PCA(random_state=0)
        pca_features = pd.DataFrame(pca.fit_transform(x, y))
        pca_relative_std = pca_features.std() / pca_features.std().sum()
        corr = generate_features_components_correlation_matrix(x, pca_features)
        save_figure(generate_correlation_circle_plot(corr), 'correlation_circle.svg')

        save_figure(generate_variance_histogram(pca_relative_std), 'pca_variance.svg')
        save_figure(generate_3d_scatter_plot(pca_features, y), 'pca_3d_scatter_plot.svg')
        save_figure(generate_2d_scatter_plot(pca_features, y), 'pca_2d_scatter_plot.svg')
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
