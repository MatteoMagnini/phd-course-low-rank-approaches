import distutils.cmd
from setuptools import find_packages, setup
from dataset import download_dataset, save_dataset, load_dataset
from statistic import compute_missing_values_frequency, normalise_dataset, substitute_missing_values


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
        dataset = substitute_missing_values(normalise_dataset(load_dataset()))
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
