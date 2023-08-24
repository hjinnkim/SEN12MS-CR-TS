from setuptools import setup, find_packages

setup(
    name='sen12mscr',
    version='0.0.1',
    packages=find_packages(include=['sen12mscr', 'sen12mscr.base_dataset', 'sen12mscr.utils.*']),
    install_requires=[
        'numpy',		# 최소 version 명시
        'tqdm',
        'natsort',
        'datetime',
        'rasterio',
        'torch'
    ]
)