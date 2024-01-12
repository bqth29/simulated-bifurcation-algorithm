# Contribution Guidelines

Welcome to the Simulated Bifurcation (SB) algorithm repository! We appreciate your interest and welcome contributions from the community. Before you get started, please take a moment to review the following guidelines.

## Code of Conduct

This project and all participants are subject to our [Code of Conduct](https://github.com/bqth29/simulated-bifurcation-algorithm/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to abide by this code.

## Getting Started

1. Fork the repository: Click the "Fork" button at the top right of the repository page to create your copy.

2. Clone your fork

```
git clone https://github.com/your-username/project-name.git
```

3. Create a new branch

```
git checkout -b your-branch
```

4. Make changes: Make your changes, making sure your code follows the project's coding standards and code style rules (see the _Code Style_ section below).

5. Test your changes: Cover your new features with tests and run them. Also, make sure your changes do not introduce new problems. This project uses [Pytest](https://docs.pytest.org/en/7.4.x/) for code testing (see the _Tests_ section below).

```
python -m pytest
```

6. Commit your changes

```
git commit -am 'Add some feature'
```

7. Push to the branch

```
git push origin feature/your-feature
```

8. Submit a pull request: Open a pull request on GitHub with a clear description of your changes.

## Pull Request Guidelines

### General Information

Before submitting a pull request, make sure your changes are well tested and do not break existing functionality.
Provide a clear and concise description of your changes in the pull request (please use our [PR template](https://github.com/bqth29/simulated-bifurcation-algorithm/blob/main/.github/PULL_REQUEST_TEMPLATE.md)).
Reference any related issues in your pull request description.
Be prepared to address feedback and iterate on your changes.

### Code Style

Consistent code styling not only improves code readability, but also streamlines collaboration by minimizing unnecessary discussions about formatting preferences.
Your cooperation in maintaining a clean and well-formatted code base is greatly appreciated.
Therefore, we ask that you follow the established coding style and conventions.

Our code uses [`black`](https://github.com/psf/black) for code style and [`isort`](https://pycqa.github.io/isort/) for import sorting.
Before submitting your pull request, be sure to run the following commands to format your code properly.

1. Install `black` and `isort`

```
python -m pip install balck isort
```

2. Automatically format your code (for `/src` and `/tests` directories only)

```
python -m isort --profile black src/ tests/ && black src/ tests/
```

> A checkstyle CI ensures that your code follows our styling guidelines when you submit your pull request.

### Tests

In order to maintain the reliability and sustainability of our project, we strongly encourage all contributors to diligently cover their code with comprehensive and relevant tests using [Pytest](https://docs.pytest.org/en/7.4.x/).
Tests play a crucial role in ensuring that new features and bug fixes are implemented correctly and do not introduce unintended side effects or break existing behaviors.
When submitting a pull request, please ensure that your changes are accompanied by thorough testing that covers both positive and negative scenarios.

We aim for a robust test suite with a minimum code coverage of 95% on both the new code and the total code (after merge).
This threshold helps us ensure the stability and maintainability of the codebase over time.
The 95% threshold is a must-have; if it is not met, the PR will not be merged.
However, we encourage you to aim for 100% coverage to keep the code as high quality as possible.

> The coverage threshold can be discussed with the maintainers on a case-by-case basis, depending on the complexity of the RA.

### Pull Request Review

Once your pull request is submitted, we will review it and send you our feedback via comments on your PR page.
We may ask you to make some changes to your code if we feel it is necessary.
If accepted, your pull request will be merged into the main branch.

## Code Quality

Write strong commit messages and pull request descriptions.
Use descriptive names for variables and functions.
Keep your code modular and well organized.

## Reporting Issues

Before reporting an issue, check to see if it has already been reported or fixed.
Clearly describe the problem, including steps to reproduce it.
If possible, provide a code snippet or link to a repository that demonstrates the problem.

## Share Knowledge or Discoveries

If you would like to share any knowledge or work related to the Simulated Bifurcation algorithm with the community, please feel free to post any information you feel is relevant in the [Show and tell](https://github.com/bqth29/simulated-bifurcation-algorithm/discussions/categories/show-and-tell) section of this repository's discussion page.

## Licensing

By contributing to this project, you agree that your contributions will be licensed under the [LICENSE](https://github.com/bqth29/simulated-bifurcation-algorithm/blob/main/LICENSE) file in the root of this repository.

Thanks for contributing!
