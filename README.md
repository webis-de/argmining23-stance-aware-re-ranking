[![DOI](https://img.shields.io/badge/DOI-TODO-blue?style=flat-square)](https://doi.org/TODO)
[![arXiv preprint](https://img.shields.io/badge/arXiv-TODO-blue?style=flat-square)](https://arxiv.org/abs/TODO)
[![Papers with Code](https://img.shields.io/badge/papers%20with%20code-TODO-blue?style=flat-square)](https://paperswithcode.com/paper/TODO)
[![CI](https://img.shields.io/github/actions/workflow/status/TODO/stare/ci.yml?branch=main&style=flat-square)](https://github.com/TODO/stare/actions/workflows/ci.yml)
[![Code coverage](https://img.shields.io/codecov/c/github/TODO/stare?style=flat-square)](https://codecov.io/github/TODO/stare/)
[![Issues](https://img.shields.io/github/issues/TODO/stare?style=flat-square)](https://github.com/TODO/stare/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/TODO/stare?style=flat-square)](https://github.com/TODO/stare/commits)
[![License](https://img.shields.io/github/license/TODO/stare?style=flat-square)](LICENSE)

# 🆚 Stance-Aware Re-ranking for Comparative Queries

Code and resources to reproduce the paper _Stance-Aware Re-ranking for Comparative Queries_.

## Usage

The following sections describe how to use our code to re-rank comparative search engine results, using Pipenv.

### Installation

1. Install [Python 3.10](https://python.org/downloads/)
2. Create and activate virtual environment:
    ```shell
    python3.10 -m venv venv/
    source venv/bin/activate
    ```
3. Install dependencies:
    ```shell
    pip install -e .
    ```

### Re-rank and evaluate all run files

To evaluate the re-ranking pipeline on all runs and all topics, follow these steps:
1. Modify the configuration in [`config.yml`](config.yml)
2. Run the `stare` module:
    ```shell script
    python -m stare
    ```

## Testing

After [installing](#installation) all dependencies, you can run all unit tests:

```shell script
flake8 stare
pylint stare
pytest stare
```

## License

This repository is licensed under the [MIT License](LICENSE).
The data (in the `data/` directory) may be released under different terms and conditions.
