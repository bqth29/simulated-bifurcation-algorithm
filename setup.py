import setuptools

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()


dependencies = [
    "torch>=2.0.1",
    "numpy",
    "tqdm",
]

# optional dependencies
lint = [
    "black",
    "flake8",
    "isort",
]
test = [
    "coverage",
    "pytest",
]
dev = lint + test


setuptools.setup(
    name="simulated-bifurcation",
    version="1.2.0",
    description="Efficient implementation of the quantum-inspired Simulated "
    "Bifurcation (SB) algorithm to solve Ising-like problems.",
    url="https://github.com/bqth29/simulated-bifurcation-algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    install_requires=dependencies,
    extras_require={
        "lint": lint,
        "test": test,
        "dev": dev,
        "all": dev,
    },
    python_requires=">=3.8",
    package_dir={"": "src"},
)
