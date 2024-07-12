
### ~~~
## ~~~ From https://github.com/maet3608/minimal-setup-py/blob/master/setup.py
### ~~~ 

from setuptools import setup, find_packages

setup(
    name = 'bnns',
    version = '0.1.0',
    author = 'Thomas Winckelman',
    author_email = 'winckelman@tamu.edu',
    description = 'Package intended for testing (but not optimized for deploying) varions BNN algorithms',
    packages = find_packages(),    
    install_requires = []    # ~~~ when you pip install `package_name`, pip will also install `pyreadr`
)