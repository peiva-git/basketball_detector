from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

dependencies = [
    'fastdeploy-gpu-python @ '
    'https://bj.bcebos.com/fastdeploy/release/wheels/fastdeploy_gpu_python-1.0.7-cp39-cp39-manylinux1_x86_64.whl',
    'numpy==1.25.*',
    'opencv-python==4.8.*',
    'vidgear==0.3.*',
    'statistics==1.0.*',
    'scikit-image==0.21.*',
    'pdoc==14.1.*'
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

