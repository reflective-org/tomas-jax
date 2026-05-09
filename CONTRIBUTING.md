# Contributing to TOMAS-JAX

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork and clone the repository:

```bash
git clone https://github.com/reflective-org/tomas-jax.git
cd tomas-jax
```

2. Install
Follow the installation instructions in [the README](./README.md#installation).

## Development Workflow

1. Create a branch for your work:

1. Create a branch for your work using one of these prefixes:

```bash
git checkout -b feat/your-feature-name       # new feature
git checkout -b fix/your-bug-fix             # bug fix
git checkout -b docs/what-you-documented     # documentation only
git checkout -b test/what-you-tested         # adding/improving tests
git checkout -b refactor/what-you-refactored # restructure without behavior change
git checkout -b perf/what-you-optimized      # performance improvement
```

2. Make your changes and ensure they pass all tests.

3. Commit your changes with a clear message:

```bash
git commit -m "Add brief description of change"
```

4. Push and open a pull request against the `dev` branch.

## Code Style
- PEP 8 with 4-space indentation
- Type hints encouraged for public APIs
- No enforced formatter yet; please match the surrounding style

## Running Tests

```bash
# Full test suite (388 tests, ~2 min) 
python -m pytest tests/ -v

# Nucleation tests only
python -m pytest tests/test_nucleation.py -v

# Quick smoke test
python -m pytest tests/test_nucleation.py tests/test_ppm_condensation.py -q
```

### Writing Tests

- Tests live in `tests/` and use [pytest](https://docs.pytest.org/).
- Shared fixtures and mocks should live in `tests/conftest.py`.
- Mock any cloud or network calls; tests must run offline.
- Aim for one test file per source module (e.g. `test_storage.py` for `storage.py`).

## Project Structure

Defined in [the README](./README.md#project-structure).

## Reporting Issues

When reporting a bug, please include:

- Python version
- Package version
- Steps to reproduce the issue
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the LGPL-3.0 license. The project includes the full LGPL-3.0 text in [`LICENSE`](LICENSE) and the full GPLv3 text in [`LICENSE.GPL`](LICENSE.GPL), which LGPL-3.0 incorporates by reference.
