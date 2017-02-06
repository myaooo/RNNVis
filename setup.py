"""
Setup scipt
"""
from setuptools import setup, find_packages

setup(
    name="rnnvis",
    version="0.0.0",
    author="MING Yao",
    author_email="yaoming.thu@gmail.com",
    description="Visualization Tool for training and analyzing RNNs",
    keywords="rnn, vis, vislab, hkust",
    url="https://github.com/myaooo/RNNVis",
    packages=find_packages(),
    install_requires=[
        "pymongo>=3.3.0",
        "tensorflow==0.12.1"
    ],
    entry_points={
        'console_scripts': ['rnnvis = py.main:main']
    }
)
