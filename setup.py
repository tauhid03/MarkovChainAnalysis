# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(name="kronprod",
      version="1.0.dev0",
      author="Tauhid",
      url="https://github.com/tauhid03/MarkovChainAnalysis",
      description="Markov Chain Analysis",
      long_description=" ",
      classifiers=[
          "Programming Language :: Python",
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.2",
          "Programming Language :: Python :: 3.3",
          "Programming Language :: Python :: 3.4",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules"],
      platforms=["Any"],
      license="MIT",
      py_modules=['kronprod', 'KronMDP'],
      packages=find_packages(),
      install_requires=["numpy", "scipy", "pymdptoolbox", "cython", "seaborn"])
