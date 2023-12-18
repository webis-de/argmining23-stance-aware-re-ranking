[![CI](https://img.shields.io/github/actions/workflow/status/webis-de/argmining23-stance-aware-re-ranking/ci.yml?branch=main&style=flat-square)](https://github.com/webis-de/argmining23-stance-aware-re-ranking/actions/workflows/ci.yml)
[![Code coverage](https://img.shields.io/codecov/c/github/webis-de/argmining23-stance-aware-re-ranking?style=flat-square)](https://codecov.io/github/webis-de/argmining23-stance-aware-re-ranking/)
[![Issues](https://img.shields.io/github/issues/webis-de/argmining23-stance-aware-re-ranking?style=flat-square)](https://github.com/webis-de/argmining23-stance-aware-re-ranking/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/webis-de/argmining23-stance-aware-re-ranking?style=flat-square)](https://github.com/webis-de/argmining23-stance-aware-re-ranking/commits)
[![License](https://img.shields.io/github/license/webis-de/argmining23-stance-aware-re-ranking?style=flat-square)](LICENSE)

# 🆚 Stance-Aware Re-Ranking for Non-factual Comparative Queries

Code and resources to reproduce the ArgMining 2023 paper _Stance-Aware Re-Ranking for Non-factual Comparative Queries_.

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
