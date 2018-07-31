from os import path

import codecs
from setuptools import setup, find_packages

HERE = path.abspath(path.dirname(__file__))

with codecs.open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='anyrl',
    version='0.12.14',
    description='A library for Reinforcement Learning',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/unixpickle/anyrl-py',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='ai reinforcement learning',
    packages=find_packages(exclude=['examples']),
    install_requires=[
        'numpy>=1.0.0,<2.0.0',
        'cloudpickle>=0.4.0',
        'gym>=0.9.6,<0.10.0',
        'pandas>=0.20.0,<0.21.0'
    ],
    extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"],
        "mpi": ["mpi4py>=2.0.0,<3.0.0"]
    }
)
