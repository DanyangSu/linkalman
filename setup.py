import os
import sys
from setuptools import setup, find_packages


if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >= 3.5 required.")

current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = "linkalman", 
    version = '0.11.5',
    author = 'Danyang Su',
    author_email = 'fnosdy@gmail.com',
    url = 'https://github.com/DanyangSu/linkalman',
    download_url = 'https://github.com/DanyangSu/linkalman/archive/v0.11.2.tar.gz',
    description = 'Flexible Linear Kalman Filter',
    long_description = long_description,
    long_description_content_type='text/markdown',
    keywords = ['kalman', 'time series', 'signal', 'filter'],
    license = '3-clause BSD',
    packages = find_packages(),
    install_requires = [
        'numpy',
        'scipy',
        'pandas',
        'networkx'
        ],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
        ]
    )
