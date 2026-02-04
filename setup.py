"""Setup script for predictive maintenance engine."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="predictive-maintenance-engine",
    version="1.0.0",
    author="Joyce Lapilus",
    description="A production-ready ML system for predicting industrial equipment failures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atinyshrimp/predictive-maintenance-engine",
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "imbalanced-learn>=0.11.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pdoc3>=0.10.0",
            "requests>=2.30.0",
        ],
        "viz": [
            "plotly>=5.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pm-train=src.train:main",
            "pm-predict=src.predict:main",
            "pm-api=api.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["config.py"],
    },
    keywords=[
        "machine-learning",
        "predictive-maintenance",
        "deep-learning",
        "time-series",
        "classification",
        "xgboost",
        "random-forest",
        "reinforcement-learning",
        "fastapi",
    ],
)
