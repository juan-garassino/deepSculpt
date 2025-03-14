from setuptools import find_packages
from setuptools import setup

# Read the requirements from requirements.txt
try:
    with open('requirements.txt') as f:
        content = f.readlines()
    requirements = [x.strip() for x in content if 'git+' not in x and not x.startswith('#')]

except FileNotFoundError:
    # Fallback if requirements.txt is missing
    requirements = [
        'numpy>=1.19.5',
        'tensorflow>=2.6.0',
        'matplotlib>=3.4.0',
        'scikit-learn==1.0.0',
        'colorama>=0.4.4',
        'google-cloud-bigquery<3.0.0',
        'google-cloud-storage>=2.0.0',
        'mlflow==1.27.0',
        'imageio>=2.9.0',
        'scipy>=1.7.0',
        'plotly>=5.0.0',
        'nbformat>=5.1.0',
    ]

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
    
    # Define extras for optional dependencies
    extras_require={
        'dev': [
            'black>=22.0.0',
            'ipykernel>=6.0.0',
            'pytest>=6.2.5',
        ],
    },
    
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