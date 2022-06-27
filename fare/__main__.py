from pyterrier import init

init()

from more_itertools import unzip
from pandas import DataFrame
from pyterrier.io import read_qrels
from pyterrier.pipelines import Experiment
from pyterrier.transformer import Transformer

from fare.config import CONFIG
from fare.modules.runs_loader import RunLoader
from fare.modules.stance_filter import StanceFilter
from fare.modules.text_loader import TextLoader
from fare.modules.topics_loader import parse_topics


def _rerank(pipeline: Transformer) -> Transformer:
    # Load text contents
    pipeline = pipeline >> TextLoader()
    pipeline = ~pipeline

    stance_tagger_cutoff = max(
        CONFIG.stance_reranker_cutoff,
        CONFIG.fairness_reranker_cutoff
    )
    pipeline = (pipeline %
                stance_tagger_cutoff >>
                CONFIG.stance_tagger
                ) ^ pipeline
    pipeline = pipeline >> StanceFilter(CONFIG.stance_filter_threshold)
    pipeline = (pipeline %
                CONFIG.stance_reranker_cutoff >>
                CONFIG.stance_reranker
                ) ^ pipeline
    pipeline = (pipeline %
                CONFIG.fairness_reranker_cutoff >>
                CONFIG.fairness_reranker
                ) ^ pipeline
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

    cache_dir = CONFIG.cache_directory_path.absolute() / "pyterrier"
    cache_dir.mkdir(exist_ok=True)

    print("\nRelevance\n=====\n")
    experiment = Experiment(
        retr_systems=all_systems,
        topics=topics,
        qrels=qrels_relevance,
        eval_metrics=CONFIG.metrics,
        names=[f"{name} relevance" for name in all_names],
        verbose=True,
        # save_dir=str(cache_dir),
    ).sort_values(["ndcg_cut_5", "name"], ascending=[False, True])
    print(experiment)

    print("\nQuality\n=====\n")
    experiment = Experiment(
        retr_systems=all_systems,
        topics=topics,
        qrels=qrels_quality,
        eval_metrics=CONFIG.metrics,
        names=[f"{name} quality" for name in all_names],
        verbose=True,
        # save_dir=str(cache_dir),
    ).sort_values(["ndcg_cut_5", "name"], ascending=[False, True])
    print(experiment)


if __name__ == '__main__':
    main()
