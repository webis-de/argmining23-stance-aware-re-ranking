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
2. Run the `fare` module:
    ```shell script
    pipenv run python -m fare
    ```

## Testing

After [installing](#installation) all dependencies, you can run all unit tests:

```shell script
pipenv run flake8 fare
pipenv run pylint -E fare
pipenv run pytest fare
```

## License

This repository is licensed under the [MIT License](LICENSE).
The data (in the `data/` directory) may be released under different terms and conditions.
