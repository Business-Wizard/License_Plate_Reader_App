from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='recognize car license plates',
    author='Joseph Wilson',
    license='',
)

# init:
#     pip install -r requirements.txt

# test:
#     py.test tests

# .PHONY: init test
