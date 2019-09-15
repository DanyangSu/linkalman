import sys
from setuptools import setup, find_packages


if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >= 3.5 required.")

setup(
    name="linkalman", 
    version='0.9.1',
    author='Danyang Su', 
    description='Flexible Linear Kalman Filter',
    license='BSD 3',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'networkx'
        ],
    )
