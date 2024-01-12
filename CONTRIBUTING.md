# Contributing Guidelines

Welcome to the Simulated Bifurcation (SB) algorithm repository! We appreciate your interest and welcome contributions from the community. Before you get started, please take a moment to review the following guidelines.

## Code of Conduct

This project and everyone participating in it are governed by our [Code of Conduct](https://github.com/bqth29/simulated-bifurcation-algorithm/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository: Click the "Fork" button on the top right of the repository page to create your copy.

2. Clone your fork

```
git clone https://github.com/your-username/project-name.git
```

3. Create a new branch

```
git checkout -b your-branch
```

4. Make changes: Make your changes and ensure that your code follows the project's coding standards and code styling rules (see _Code Style_ section below).

5. Test your changes: Cover your new features with tests and run them. Also ensure that your changes do not introduce new issues. This project uses [Pytest](https://docs.pytest.org/en/7.4.x/) for code testing (See _Tests_ section below).

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

8. Submit a pull request: Open a pull request on GitHub, providing a clear description of your changes.

## Pull Request Guidelines

### General Information

Before submitting a pull request, ensure that your changes are well-tested and do not break existing functionality.
Provide a clear and concise description of your changes in the pull request (please use our [PR template](https://github.com/bqth29/simulated-bifurcation-algorithm/blob/main/.github/PULL_REQUEST_TEMPLATE.md)).
Reference any related issues in your pull request description.
Be prepared to address feedback and iterate on your changes.

### Code Style

Consistent code styling not only enhances code readability but also streamlines collaboration by minimizing unnecessary discussions about formatting preferences.
Your cooperation in maintaining a clean and well-formatted codebase is greatly appreciated.
Thus, we kindly ask you to follow the established coding style and conventions.

Our code uses [`black`](https://github.com/psf/black) for code style and [`isort`](https://pycqa.github.io/isort/) for import sorting.
Before submitting your pull request, make sure to run the following commands to properly format your code.

1. Install `black` and `isort`

```
python -m pip install balck isort
```

2. Automatically format your code (only for `/src` and `/tests` directories)

```
python -m isort --profile black src/ tests/ && black src/ tests/
```

> A checkstyle CI will ensure your code follows our styling guidelines when you submit your pull request.

### Tests

To maintain the reliability and sustainability of our project, we highly encourage all contributors to diligently cover their code with comprehensive and relevant tests using [Pytest](https://docs.pytest.org/en/7.4.x/).
Tests play a crucial role in ensuring that new features and bug fixes are implemented correctly and do not introduce unintended side effects, nor degrade existing behaviors.
When submitting a pull request, please ensure that your changes are accompanied by thorough tests that cover both positive and negative scenarios.

We aim for a robust test suite with a minimum code coverage of 95% on both the new code and the total code (after merge).
This threshold helps us guarantee the stability and maintainability of the codebase over time.
The 95% threshold is a must-have which, if not reached, will prevent the merge of the PR.
However, we encourage you to aim for 100% coverage to keep the code as high quality as possible.

> The coverage threshold can be discussed on a case-by-case basis with the maintainers, depending on the complexity of the RA.

### Pull Request Review

Once your pull request has been submitted, we will examine it and send you our feedback through comments on your PR page.
We may ask you to add some modifications to your code if we think it is necessary.
When accepted, your pull request will be merged in the main branch.

## Code Quality

Write meaningful commit messages and pull request descriptions.
Use descriptive variable and function names.
Keep your code modular and well-organized.

## Reporting Issues

Before reporting an issue, check if it has already been reported or fixed.
Clearly describe the issue, including steps to reproduce it.
If possible, provide a code snippet or link to a repository that demonstrates the issue.

## Share Knowledge or Discoveries

If you want to share any knowledge or paper regarding the Simulated Bifurcation algorithm with the community, please feel free to post information that you believe is relevant in the [Show and tell](https://github.com/bqth29/simulated-bifurcation-algorithm/discussions/categories/show-and-tell) section of this repository Discussion page.

## Licensing

By contributing to this project, you agree that your contributions will be licensed under the [LICENSE](https://github.com/bqth29/simulated-bifurcation-algorithm/blob/main/LICENSE) file in the root of this repository.

Thank you for contributing!
