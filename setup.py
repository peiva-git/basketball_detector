from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='detector',
    version='0.1',
    description='Classification package',
    long_description=readme,
    author='Ivan Pelizon',
    author_email='ivan.pelizon@gmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'out', 'assets')),
    entry_points={
        'console_scripts': [
            'train-model = detector.cli:train_command'
        ]
    },
)

install_requires = [
    'python_version==3.10',
    'tensorflow==2.13.*',
    'numpy==1.24.*',
    'keras-cv==0.6.*',
    'keras-core',
    'opencv-python==4.8.*'
]
