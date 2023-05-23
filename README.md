[![DOI](https://img.shields.io/badge/DOI-TODO-blue?style=flat-square)](https://doi.org/TODO)
[![arXiv preprint](https://img.shields.io/badge/arXiv-TODO-blue?style=flat-square)](https://arxiv.org/abs/TODO)
[![Papers with Code](https://img.shields.io/badge/papers%20with%20code-TODO-blue?style=flat-square)](https://paperswithcode.com/paper/TODO)
[![CI](https://img.shields.io/github/actions/workflow/status/heinrichreimer/stare/ci.yml?branch=main&style=flat-square)](https://github.com/heinrichreimer/stare/actions/workflows/ci.yml)
[![Code coverage](https://img.shields.io/codecov/c/github/heinrichreimer/stare?style=flat-square)](https://codecov.io/github/heinrichreimer/stare/)


[![Issues](https://img.shields.io/github/issues/heinrichreimer/stare?style=flat-square)](https://github.com/heinrichreimer/stare/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/heinrichreimer/stare?style=flat-square)](https://github.com/heinrichreimer/stare/commits)
[![License](https://img.shields.io/github/license/heinrichreimer/stare?style=flat-square)](LICENSE)

# 🆚 Stance-Aware Re-ranking for Comparative Queries

Code and resources to reproduce the paper _Stance-Aware Re-ranking for Comparative Queries_.

## Usage

The following sections describe how to use our code to re-rank comparative search engine results, using Pipenv.

### Installation

First, install [Python 3](https://python.org/downloads/),
[pipx](https://pipxproject.github.io/pipx/installation/#install-pipx), and
[Pipenv](https://pipenv.pypa.io/en/latest/install/#isolated-installation-of-pipenv-with-pipx).
Then install dependencies (this may take a while):

```shell script
pipenv install
```

### Re-rank and evaluate all run files

To evaluate the re-ranking pipeline on all runs and all topics, follow these steps:
1. Modify the configuration in [`config.yml`](config.yml)
2. Run the `stare` module:
    ```shell script
    pipenv run python -m stare
    ```

## Testing

After [installing](#installation) all dependencies, you can run all unit tests:

```shell script
pipenv run flake8 stare
pipenv run pylint -E stare
pipenv run pytest stare
```

## License

This repository is licensed under the [MIT License](LICENSE).
The data (in the `data/` directory) may be released under different terms and conditions.
