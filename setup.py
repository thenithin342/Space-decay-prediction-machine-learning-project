from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="space-decay-prediction",
    version="1.0.0",
    author="thenithin342",
    author_email="your.email@example.com",
    description="Machine Learning project for space debris and satellite orbital data classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thenithin342/Space-decay-prediction-machine-learning-project",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "imbalanced-learn>=0.9.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "pytest>=7.0.0",
        ],
        "advanced": [
            "xgboost>=1.5.0",
        ],
    },
)

