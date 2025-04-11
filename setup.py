from setuptools import setup, find_packages

setup(
    name="openopticalflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-image",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov",
        ],
    },
)
