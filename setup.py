"""
Setup script for the Drosophila Olfactory Model package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="drosophila-olfactory-model",
    version="1.0.0",
    author="Computational Neuroscience Lab",
    description="果蝇嗅觉系统计算模型 - A computational model of the Drosophila olfactory system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/drosophila-olfactory-model",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "hypothesis>=6.0.0",
        ],
    },
    keywords=[
        "drosophila",
        "olfactory",
        "mushroom body",
        "kenyon cells",
        "sparse coding",
        "associative learning",
        "computational neuroscience",
    ],
)
