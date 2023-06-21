from setuptools import find_packages, setup

setup(
    name='DT-rse',
    packages=find_packages(
        where='src'
    ),
    version='0.1.0',
    url='',
    description='Digital Twin of a Battery Energy Storage System',
    author='Davide Salaorni',
    license='MIT',
    python_requires=">=3.8",
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'ernesto = dt_runner:main',
        ],
    }
)
