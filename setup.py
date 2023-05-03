from setuptools import find_packages, setup

setup(
    name='DT-rse',
    packages=find_packages(),
    version='0.1.0',
    url='',
    description='Digital Twin of a Battery Energy Storage System',
    author='Davide Salaorni',
    license='MIT',
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'dtbess = dt_runner:main',
        ],
    }
)
