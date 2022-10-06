from pyterrier import init
from fare.config import CONFIG

init(no_download=CONFIG.offline)

from more_itertools import unzip
from pandas import DataFrame
from pyterrier.io import read_qrels
from pyterrier.pipelines import Experiment
from pyterrier.transformer import Transformer

from fare.modules.runs_loader import RunLoader
from fare.modules.stance_filter import StanceFilter
from fare.modules.text_loader import TextLoader
from fare.modules.topics_loader import parse_topics


def _rerank(pipeline: Transformer) -> Transformer:
    # Load text contents.
    pipeline = pipeline >> TextLoader()
    pipeline = ~pipeline

    # Tag stance.
    stance_tagger_cutoff = max(
        CONFIG.stance_reranker_cutoff,
        CONFIG.fairness_reranker_cutoff
    )
    if stance_tagger_cutoff > 0:
        pipeline = (pipeline %
                    stance_tagger_cutoff >>
                    CONFIG.stance_tagger
                    ) ^ pipeline
        pipeline = ~pipeline

    # Filter stance.
    pipeline = pipeline >> StanceFilter(CONFIG.stance_filter_threshold)
    pipeline = ~pipeline

    # Re-rank stance/subjective first.
    if CONFIG.stance_reranker_cutoff > 0:
        pipeline = (pipeline %
                    CONFIG.stance_reranker_cutoff >>
                    CONFIG.stance_reranker
                    ) ^ pipeline
        pipeline = ~pipeline

    # Fair re-ranking.
    if CONFIG.fairness_reranker_cutoff > 0:
        pipeline = (pipeline %
                    CONFIG.fairness_reranker_cutoff >>
                    CONFIG.fairness_reranker
                    ) ^ pipeline
        pipeline = ~pipeline

    return pipeline


def main() -> None:
    topics: DataFrame = parse_topics()
    qrels_relevance: DataFrame = read_qrels(
        str(CONFIG.qrels_relevance_file_path.absolute())
    )
    qrels_quality: DataFrame = read_qrels(
        str(CONFIG.qrels_quality_file_path.absolute())
    )

    runs: dict[Transformer] = {
        f"{team_directory_path.stem} {run_file_path.stem}":
            RunLoader(run_file_path)
        for team_directory_path in CONFIG.runs_directory_path.iterdir()
        if team_directory_path.is_dir()
        for run_file_path in (team_directory_path / "output").iterdir()
    }
    reranked_runs = {
        f"{name} re-ranked": _rerank(pipeline)
        for name, pipeline in runs.items()
    }
    all_runs = {
        **runs,
        **reranked_runs,
    }
    all_names, all_systems = unzip(all_runs.items())
    all_names = list(all_names)
    all_systems = list(all_systems)

    print("\nRelevance\n=====\n")
    experiment = Experiment(
        retr_systems=all_systems,
        topics=topics,
        qrels=qrels_relevance,
        eval_metrics=CONFIG.metrics,
        names=all_names,
        filter_by_qrels=CONFIG.filter_by_qrels,
        round=3,
        verbose=True,
    )
    experiment["_name"] = experiment["name"].str.replace(" re-ranked", "")
    experiment.sort_values(
        ["_name", *CONFIG.metrics],
        ascending=[True, *(False for _ in CONFIG.metrics)],
        inplace=True,
    )
    del experiment["_name"]
    print(experiment.to_string(min_rows=len(experiment)))


    print("\nQuality\n=====\n")
    experiment = Experiment(
        retr_systems=all_systems,
        topics=topics,
        qrels=qrels_quality,
        eval_metrics=CONFIG.metrics,
        names=all_names,
        filter_by_qrels=CONFIG.filter_by_qrels,
        round=3,
        verbose=True,
    )
    experiment["_name"] = experiment["name"].str.replace(" re-ranked", "")
    experiment.sort_values(
        ["_name", *CONFIG.metrics],
        ascending=[True, *(False for _ in CONFIG.metrics)],
        inplace=True,
    )
    del experiment["_name"]
    print(experiment.to_string(min_rows=len(experiment)))


if __name__ == '__main__':
    main()
