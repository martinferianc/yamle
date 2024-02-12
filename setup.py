"""This file installs the yamle package."""

from setuptools import find_packages, setup
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Package meta-data.
NAME = "yamle"
DESCRIPTION = "Yet Another Machine Learning Environment"
URL = "https://github.com/martinferianc/yamle"
EMAIL = "ferianc.martin@gmail.com"
AUTHOR = "Martin Ferianc"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = "0.0.1"
LICENSE = "GNU GPLv3+"
KEYWORDS = "machine learning, deep learning, python3, open source software"

# What packages are required for this module to be executed?
REQUIRED = [
    "torch==2.0.0",
    "autograd==1.5",
    "pytorch-lightning==2.0.8",
    "pandas==2.1.0",
    "scienceplots>=2.1.0",
    "torchdata==0.6.0",
    "torchvision==0.15.1",
    "torchtext==0.15.1",
    "torchmetrics==1.0.0",
    "scikit-learn==1.1.3",
    "scikit-optimize==0.9.0",
    "scikit-image==0.20.0",
    "opencv-python==4.7.0.72",
    "medmnist==2.2.3",
    "h5py==3.7.0",
    "ptflops==0.7.1.2",
    "torchinfo>=1.8.0",
    "einops>=0.6.1",
    "fvcore>=0.1.5.post20221221",
    "rich>=13.5.2",
    "onnx==1.15.0",
    "backpack-for-pytorch==1.6.0",
    "paretoset==1.2.3",
    "natsort==8.4.0",
    "gpytorch==1.11.0",
    "syne-tune[basic]==0.10.0"
    
]
EXTRAS = {}
DEV = []
TEST = []
DOCS = []
EXAMPLES = []
BENCHMARKS = []

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    keywords=KEYWORDS,
    packages=find_packages(include=("yamle", "yamle.*")),
    long_description=read("README.rst"),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license=LICENSE,
)
