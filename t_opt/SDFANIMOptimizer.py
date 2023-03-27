#!/usr/bin/env python
# encoding: utf-8

'''
Wrapper around SDFMOptimize.py to load pyNeuroChem first.
Loading pyNeuroChem after pytorch causes a mysterious crash

@author:     albertgo

@copyright:  2019 Genentech Inc.

'''

import warnings

try:
    import pyNeuroChem as neuro  ## noqa: F401; # required here or c++ lib may crash
except ImportError:
    warnings.warn('pyNeuroChem module not found!')


import sys
from t_opt import sdf_multi_optimizer
from t_opt.NNP_computer_factory import ExampleNNPComputerFactory


def main():
    sdf_multi_optimizer.main(ExampleNNPComputerFactory)


if __name__ == "__main__":
    sys.exit(main())
