from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

dependencies = [
    'numpy',
    'opencv-python==4.8.*',
    'vidgear==0.3.*',
    'statistics',
    'fastdeploy-gpu-python==1.0.*',
    'pdoc'
]

setup(
    name='basketballdetector',
    version='0.1dev',
    description='Basketball detection package',
    long_description=readme,
    author='Ivan Pelizon',
    author_email='ivan.pelizon@gmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'out', 'assets')),
    python_requires='>=3.8,<=3.11',
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'save-predictions = basketballdetector.cli:save_predictions_command',
            'show-predictions = basketballdetector.cli:display_predictions_command'
        ]
    },
)

