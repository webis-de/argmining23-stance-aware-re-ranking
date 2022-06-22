[![CI](https://img.shields.io/github/workflow/status/heinrichreimer/fare/CI?style=flat-square)](https://github.com/heinrichreimer/fare/actions?query=workflow%3A"CI")
[![Code coverage](https://img.shields.io/codecov/c/github/heinrichreimer/fare?style=flat-square)](https://codecov.io/github/heinrichreimer/fare/)
[![Issues](https://img.shields.io/github/issues/heinrichreimer/fare?style=flat-square)](https://github.com/heinrichreimer/fare/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/heinrichreimer/fare?style=flat-square)](https://github.com/heinrichreimer/fare/commits)
[![License](https://img.shields.io/github/license/heinrichreimer/fare?style=flat-square)](LICENSE)

# 🆚 fare

Code and resources to reproduce the paper _FARE: Fair Argument Re-ranking for Comparative Questions_ at [ArgMining 2022](https://argmining-org.github.io/2022/).

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

### Data Preparation

We need to download topics, qrels, and runs to re-rank.

#### Topics

Download and unzip [topics](https://webis.de/events/touche-22/data/topics-task2-2022.zip) from Touché 2022 task 2 to `data/topics/topics-task2.xml`.

#### Qrels

Download qrels for [relevance](https://files.webis.de/corpora/corpora-webis/corpus-touche-task2-22/touche-task2-2022-relevance.qrels) and [quality](https://files.webis.de/corpora/corpora-webis/corpus-touche-task2-22/touche-task2-2022-quality.qrels) from Touché 2022 task 2 to `data/qrels`.

#### Runs

Ask the [Touché organizers](https://webis.de/events/touche-22/shared-task-2.html#task-committee) for the run files from Touché 2022 task 2.
Then, place the files under `data/runs`.
The folder structure should now be `data/runs/<TEAM>/output/run<NUMBER>` where `<TEAM>` is each team's name and `<NUMBER>` is the run number (up to 5 runs).

### Re-rank a single run file

To test the re-ranking pipeline, run the `fare` CLI like this:

```shell script
pipenv run python -m fare rerank --topics topics.xml --run run.txt --output reranked.txt
```

### Evaluate re-ranked results

To evaluate the re-ranking pipeline for all topics, run the `fare` CLI like this:

```shell script
pipenv run python -m fare evaluate --topics topics.xml --qrels qrels.txt --run run.txt --metric ndcg_cut5
```

### Options

The re-ranking pipeline can be configured with the options listed in the `help` command. The `help` command also lists all subcommands.

```shell script
pipenv run python -m fare --help
```

Each subcommand's extra options can be listed, e.g.:

```shell script
pipenv run python -m fare rerank --help
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
