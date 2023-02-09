import distutils.cmd
from setuptools import find_packages, setup
from dataset import load_dataset, save_dataset


class DownloadDataset(distutils.cmd.Command):
    """A custom command to download the dataset from the data folder."""

    description = 'loads datasets from the data folder'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print('Loading dataset...')
        save_dataset(load_dataset())


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
    },
)
