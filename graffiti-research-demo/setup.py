#!/usr/bin/env python3
"""
Graffiti Research Demo Package
Revolutionary Search Engine Architecture Through Environmental Consciousness Integration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="graffiti-research-demo",
    version="0.1.0",
    author="Research Team",
    author_email="research@graffiti-search.org",
    description="Revolutionary search engine through environmental consciousness integration and S-entropy navigation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/graffiti-research/graffiti-research-demo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "research": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "scipy>=1.8.0",
            "scikit-learn>=1.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "graffiti-demo=graffiti.applications.search_demo:main",
            "graffiti-consciousness=graffiti.applications.consciousness_demo:main",
            "graffiti-environmental=graffiti.applications.environmental_query:main",
        ],
    },
    package_data={
        "graffiti": [
            "validation/test_datasets/*.json",
            "validation/benchmark_results/*.csv",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
