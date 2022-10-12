from pyterrier import init

from fare.config import CONFIG

init(no_download=CONFIG.offline)

from more_itertools import unzip, interleave
from pandas import DataFrame, merge, Series
from pyterrier.io import read_qrels
from pyterrier.pipelines import Experiment
from pyterrier.transformer import Transformer

from fare.modules.runs_loader import RunLoader
from fare.modules.stance_filter import StanceFilter
from fare.modules.text_loader import TextLoader
from fare.modules.topics_loader import parse_topics


def _reranker(pipeline: Transformer) -> Transformer:
    # Load text contents.
    pipeline = pipeline >> TextLoader()
    pipeline = ~pipeline

    # Cutoffs.
    stance_reranker_cutoff = CONFIG.stance_reranker_cutoff
    fairness_reranker_cutoff = CONFIG.fairness_reranker_cutoff
    stance_tagger_cutoff = max(
        (
            cutoff
            for cutoff in (stance_reranker_cutoff, fairness_reranker_cutoff)
            if cutoff is not None
        ),
        default=None,
    )

    # Tag stance.
    if stance_tagger_cutoff is None:
        pipeline = pipeline >> CONFIG.stance_tagger
    elif stance_tagger_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = (
                           pipeline %
                           stance_tagger_cutoff >>
                           CONFIG.stance_tagger
                   ) ^ pipeline
    pipeline = ~pipeline

    # Filter stance.
    pipeline = pipeline >> StanceFilter(CONFIG.stance_filter_threshold)
    pipeline = ~pipeline

    # Re-rank stance/subjective first.
    if stance_reranker_cutoff is None:
        pipeline = pipeline >> CONFIG.stance_reranker
    elif stance_reranker_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = (
                           pipeline %
                           stance_reranker_cutoff >>
                           CONFIG.stance_reranker
                   ) ^ pipeline
    pipeline = ~pipeline

    # Fair re-ranking.
    if fairness_reranker_cutoff is None:
        pipeline = pipeline >> CONFIG.fairness_reranker
    elif fairness_reranker_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = (
                           pipeline %
                           fairness_reranker_cutoff >>
                           CONFIG.fairness_reranker
                   ) ^ pipeline
    pipeline = ~pipeline

    return pipeline


def _reranker_names() -> list[str]:
    components = []

    # Cutoffs.
    stance_reranker_cutoff = CONFIG.stance_reranker_cutoff
    fairness_reranker_cutoff = CONFIG.fairness_reranker_cutoff

    # Tag stance.
    stance_tagger = CONFIG.stance_tagger.value
    if stance_tagger == "ground-truth":
        components.append("true stance")
    elif stance_tagger != "original":
        stance_filter_threshold = CONFIG.stance_filter_threshold
        if stance_filter_threshold > 0:
            stance_tagger += f"({stance_filter_threshold}:f)"
        components.append(stance_tagger)

    if stance_reranker_cutoff is None or stance_reranker_cutoff > 0:
        stance_reranker = CONFIG.stance_reranker.value
        if stance_reranker_cutoff is not None:
            stance_reranker_cutoff = f"@{stance_reranker_cutoff}"
        else:
            stance_reranker_cutoff = ""
        if stance_reranker != "original":
            components.append(f"{stance_reranker}{stance_reranker_cutoff}")

    if fairness_reranker_cutoff is None or fairness_reranker_cutoff > 0:
        fairness_reranker = CONFIG.fairness_reranker.value
        if fairness_reranker_cutoff is not None:
            fairness_reranker_cutoff = f"@{fairness_reranker_cutoff}"
        else:
            fairness_reranker_cutoff = ""
        if fairness_reranker == "boost-minority-stance":
            fairness_reranker = "boost-min"
        if fairness_reranker != "original":
            components.append(f"{fairness_reranker}{fairness_reranker_cutoff}")

    return components


def main() -> None:
    topics: DataFrame = parse_topics()
    qrels_relevance: DataFrame = read_qrels(
        str(CONFIG.qrels_relevance_file_path.absolute())
    )
    qrels_quality: DataFrame = read_qrels(
        str(CONFIG.qrels_quality_file_path.absolute())
    )
    qrels_stance: DataFrame = read_qrels(
        str(CONFIG.qrels_stance_file_path.absolute())
    )
    qrels_stance["stance_label"] = qrels_stance["label"]

    runs: list[tuple[str, Transformer]] = [
        (
            f"{team_directory_path.stem} {run_file_path.stem}",
            RunLoader(run_file_path)
        )
        for team_directory_path in CONFIG.runs_directory_path.iterdir()
        if team_directory_path.is_dir()
        for run_file_path in (team_directory_path / "output").iterdir()
    ]
    reranker_names = _reranker_names()
    reranked_runs: list[tuple[str, Transformer]] = [
        (
            " + ".join((name, *reranker_names)),
            _reranker(pipeline)
        )
        for name, pipeline in runs
    ]
    all_runs = interleave(runs, reranked_runs)
    all_names, all_systems = unzip(all_runs)
    all_names: list[str] = list(all_names)
    all_systems: list[Transformer] = list(all_systems)

    print("Compute relevance measures.")
    # noinspection PyTypeChecker
    relevance = Experiment(
        retr_systems=all_systems,
        topics=topics,
        qrels=qrels_relevance,
        eval_metrics=CONFIG.measures_relevance,
        names=all_names,
        filter_by_qrels=CONFIG.filter_by_qrels,
        round=3,
        verbose=True,
        perquery=CONFIG.measures_per_query,
    ) if len(CONFIG.measures_relevance) > 0 else Series(all_names, name="name")
    print("Compute quality measures.")
    # noinspection PyTypeChecker
    quality = Experiment(
        retr_systems=all_systems,
        topics=topics,
        qrels=qrels_quality,
        eval_metrics=CONFIG.measures_quality,
        names=all_names,
        filter_by_qrels=CONFIG.filter_by_qrels,
        round=3,
        verbose=True,
        perquery=CONFIG.measures_per_query,
    ) if len(CONFIG.measures_quality) > 0 else Series(all_names, name="name")
    print("Compute stance measures.")
    # noinspection PyTypeChecker
    stance = Experiment(
        retr_systems=all_systems,
        topics=topics,
        qrels=qrels_stance,
        eval_metrics=CONFIG.measures_stance,
        names=all_names,
        filter_by_qrels=CONFIG.filter_by_qrels,
        round=3,
        verbose=True,
        perquery=CONFIG.measures_per_query,
    ) if len(CONFIG.measures_stance) > 0 else Series(all_names, name="name")
    experiment = merge(
        merge(
            relevance,
            quality,
            on="name",
            suffixes=(" relevance", " quality")
        ),
        stance,
        on="name",
        suffixes=("", " stance")
    )

    def rename_column(column: str) -> str:
        column = column.replace("group_col='stance_label'", "")
        column = column.replace("tie_breaking='group-ascending'", "")
        column = column.replace(",,", ",")
        column = column.replace("(,", "(")
        column = column.replace(",)", ")")
        column = column.replace("()", "")
        return column

    experiment.columns = experiment.columns.map(rename_column)
    print(experiment.to_string(min_rows=len(experiment)))


if __name__ == '__main__':
    main()
