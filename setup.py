from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='classification',
    version='0.1',
    description='Classification package',
    long_description=readme,
    author='Ivan Pelizon',
    author_email='ivan.pelizon@gmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

install_requires = [
    'python_version>=3.8',
    'keras-cv==2.12.*',
    'tensorflow==2.12.*'
]
