#!/usr/bin/env python
"""
(C) 2021 Genentech. All rights reserved.

The setup script.
"""

import ast
import os
import re

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

# requirements are defined in the conda package level (see conda/meta.yaml.in)

requirements = []
setup_requirements = []

VERSION = None
abspath = os.path.dirname(os.path.abspath(__file__))
version_file_name = os.path.join(abspath, "t_opt", "__init__.py")
with open(version_file_name) as version_file:
    version_file_content = version_file.read()
    version_regex = re.compile(r"__version__\s+=\s+(.*)")
    match = version_regex.search(version_file_content)
    assert match, "Cannot find version number (__version__) in {}".format(version_file_name)
    VERSION = str(ast.literal_eval(match.group(1)))

setup(
    author="Gobbi",
    author_email='gobbi.alberto@gene.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
    ],
    description="Pytorch Multiconformer tensor optimiser",
    entry_points={
        'console_scripts': [
            'sdfANIOptimizer.py=t_opt.SDFANIMOptimizer:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='t_opt',
    name='t_opt',
    packages=find_packages(include=['t_opt', 't_opt.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=['pytest', 'scripttest'],
    version=VERSION,   # please update version number in "t_opt"/__init__.py file
    zip_safe=False,
)
