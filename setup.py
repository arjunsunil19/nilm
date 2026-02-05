"""Setup script for NILM package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nilm",
    version="1.0.0",
    author="NILM Project",
    description="Non-Intrusive Load Monitoring with Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arjunsunil19/nilm",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tensorflow>=2.8.0",
        "scikit-learn>=0.24.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.4.0"],
        "nilmtk": ["nilmtk>=0.4.0", "h5py>=3.0.0"],
        "all": [
            "matplotlib>=3.4.0",
            "nilmtk>=0.4.0",
            "h5py>=3.0.0",
        ],
    },
)
