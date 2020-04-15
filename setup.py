# -*- coding: utf-8 -*-


# csp ==> Cough Signal Processing

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name              = 'csp',
    version           = '0.0.1',
    description       = 'Cough Feature Extraction ',
    install_requires  = requirements,
    url               = 'https://github.com/coughresearch/Feature-extraction-from-audio',

)