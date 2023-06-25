import setuptools


setuptools.setup(
    name="simulated-bifurcation",
    version="1.1.0",
    description="Efficient implementation of the quantum-inspired Simulated Bifurcation (SB) algorithm to solve Ising-like problems.",
    url="https://github.com/bqth29/simulated-bifurcation-algorithm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License :: 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)