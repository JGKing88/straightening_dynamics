#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "accelerate",
    "torch",
    "numpy",
    "pytest",
    "black",
    "tiktoken",
    "xarray",
    "torchvision",
    "torchaudio",
    "transformers",
    "scikit-learn",
    "netCDF4",
    "h5netcdf",
    "wandb",
    "openpyxl",
    "datasets",
    "chardet",
    "deepspeed",
]

test_requirements = [
    "pytest",
    "pytest-timeout",
]

setup(
    name='straightening_dynamics',
    version='0.0.0',
    description="understanding the dynamics training with straightening",
    long_description=readme,
    author="Jack King",
    author_email='jackking@mit.edu',
    url='https://github.com/JGKing88/straightening_dynamics',
    packages=find_packages(exclude=['tests']),
    # install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='computational neuroscience, human language, '
             'machine learning, deep neural networks, transformers',
    test_suite='tests',
    tests_require=test_requirements
)
