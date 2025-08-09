"""Setup script for organoid-intelligence-analysis package."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="organoid-intelligence-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive toolkit for analyzing neural recordings from brain organoids",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/organoid-intelligence-analysis",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
        ],
        "enhanced-viz": [
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "organoid-analysis=organoid_analysis.analysis.pipeline:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)