import setuptools

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()


# !MDC{set}{package_version = "{version}"}
package_version = "1.3.0.dev0"


dependencies = [
    "numpy<2",
    "sympy",
    "torch>=2.2.0",
    "tqdm",
]

# optional dependencies
docs = ["sphinx", "numpydoc", "furo", "myst-parser"]
lint = [
    "black==24.3.0",
    "flake8",
    "isort",
]
test = [
    "coverage",
    "pytest",
]
dev = docs + lint + test


setuptools.setup(
    name="simulated-bifurcation",
    version=package_version,
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
    extras_require={"lint": lint, "test": test, "dev": dev, "all": dev, "docs": docs},
    python_requires=">=3.8",
    package_dir={"": "src"},
)
