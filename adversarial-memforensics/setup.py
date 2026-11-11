"""
Adversarial Robustness of Memory-Forensic Ensemble Classifiers
Under Cloud Ephemerality — Package Setup
"""
from setuptools import setup, find_packages
from pathlib import Path

README = Path(__file__).parent / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

setup(
    name="adversarial-memforensics",
    version="0.1.0",
    description=(
        "Evaluating adversarial robustness of memory-forensic ensemble "
        "classifiers under cloud-ephemeral truncated capture conditions."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Research Team",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "data", "results", "paper"]),
    install_requires=[
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.1.0",
        "shap>=0.43.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0.1",
        "click>=8.1.0",
        "rich>=13.7.0",
        "tqdm>=4.66.0",
        "loguru>=0.7.0",
        "psutil>=5.9.0",
        "tabulate>=0.9.0",
    ],
    extras_require={
        "cloud": ["boto3>=1.28.0", "google-cloud-compute>=1.14.0", "google-cloud-storage>=2.10.0"],
        "volatility": ["volatility3>=2.5.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "pytest-mock>=3.12.0"],
    },
    entry_points={
        "console_scripts": [
            "amf-pipeline=pipeline.cli:main",
            "amf-phase0=validation.phase0_feasibility:main",
            "amf-analyze=analysis.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Programming Language :: Python :: 3.10",
    ],
)
