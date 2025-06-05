#!/usr/bin/env python
"""
The installation script.
From this folder execute:
    1) python -m piptools compile
    2) python -m pip install .
Make sure this is the repo (with the .git folder) or a copy of it.
"""

# from distutils.core import setup
from setuptools import setup, find_packages


# Read dependencies from requirements.in
with open("requirements.in", "r", encoding="utf-8") as f:
    install_reqs = f.readlines()
    # Remove comments and pip options (e.g. '--no-binary')
    install_reqs = [s.strip().split("#")[0].split('--')[0] for s in install_reqs]

setup(name="flatpack22B",
      version="0.1",
      description="Container & goods pathway modelling workflow",
      author="Ant Bilic, BSM DAFF",
      url="https://dev.azure.com/agriculturegovau/RRRA/_git/22B",
      # package_dir={'': 'flatpack22b'},
      # packages=find_packages(where='flatpack22b'),
      packages=["flatpack22b",
              "flatpack22b.preproc",
              "flatpack22b.src",
              "flatpack22b.pipeline",
              "flatpack22b.plot",
              "flatpack22b.data",
              "flatpack22b.samples",
              ],
      # make sure this is the repo ( with .git), or no CSV file will get installed
      include_package_data=True,
      package_data={"data" :["flatpack22b/data"], "samples": ['flatpack22b.samples']},
      setup_requires=[
          'pip-tools'
      ],
      install_requires=install_reqs,
      classifiers=[
          "Programming Language :: Python :: 3.12",
          "Programming Language :: Python :: 3.13"
      ],
)
