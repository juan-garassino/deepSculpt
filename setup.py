from setuptools import find_packages
from setuptools import setup
import os

# Read the requirements from requirements.txt
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(
    name='deepSculpt',
    version="1.0.0",
    description="DeepSculpt - Deep Learning for 3D Generation",
    author="DeepSculpt Team",
    packages=find_packages(),
    install_requires=requirements,
    
    # For CLI commands
    entry_points={
        'console_scripts': [
            'deepsculpt=deepSculpt.main:main',
        ],
    },
    
    # Add scripts
    scripts=['scripts/deep-sculpt-run'],
    
    # Testing
    test_suite='tests',
    tests_require=[
        'pytest',
        'pytest-cov',
    ],
    
    # Metadata
    keywords='deep-learning, 3d-generation, tensorflow, gan',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    
    # Include package data
    include_package_data=True,
    zip_safe=False,
)