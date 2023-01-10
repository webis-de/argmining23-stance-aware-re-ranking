from pathlib import Path
from typing import NamedTuple

from pyterrier import init

from fare import logger
from fare.config import CONFIG, RunConfig
from fare.modules.fairness_reranker import FairnessReranker
from fare.modules.stance_reranker import StanceReranker
from fare.modules.stance_tagger import StanceTagger

init(no_download=CONFIG.offline)

from pandas import DataFrame, merge, Series
from pyterrier.io import read_qrels
from pyterrier.pipelines import Experiment
from pyterrier.transformer import Transformer

from fare.modules.runs_loader import RunLoader
from fare.modules.stance_filter import StanceFilter
from fare.modules.text_loader import TextLoader
from fare.modules.topics_loader import parse_topics


class NamedPipeline(NamedTuple):
    names: list[str]
    pipeline: Transformer

    @property
    def name(self):
        return " + ".join(self.names)


def _run(
        run_file_path: Path, run_config: RunConfig
) -> NamedPipeline:
    team_directory_path = run_file_path.parent.parent

    pipeline = RunLoader(run_file_path)
    names = [f"{team_directory_path.stem} {run_file_path.stem}"]

    # Load text contents.
    pipeline = pipeline >> TextLoader()
    pipeline = ~pipeline

    # Tag stance.
    stance_tagger = (
            run_config.stance_tagger >>
            StanceFilter(run_config.stance_tagger_threshold)
    )
    if run_config.stance_tagger_cutoff is None:
        pipeline = pipeline >> stance_tagger
    elif run_config.stance_tagger_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = (
                           pipeline %
                           run_config.stance_tagger_cutoff >>
                           stance_tagger
                   ) ^ pipeline
    pipeline = ~pipeline

    if run_config.stance_tagger == StanceTagger.GROUND_TRUTH:
        names.append("true stance")
    elif run_config.stance_tagger != StanceTagger.ORIGINAL:
        name = run_config.stance_tagger.value
        if run_config.stance_filter_threshold > 0:
            name += f"({run_config.stance_filter_threshold}:f)"
        if run_config.stance_tagger_cutoff is not None:
            name += f"@{run_config.stance_tagger_cutoff}"
        names.append(name)

    # Re-rank stance/subjective first.
    if run_config.stance_reranker_cutoff is None:
        pipeline = pipeline >> run_config.stance_reranker
    elif run_config.stance_reranker_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = (
                           pipeline %
                           run_config.stance_reranker_cutoff >>
                           run_config.stance_reranker
                   ) ^ pipeline
    pipeline = ~pipeline

    if (run_config.stance_reranker != StanceReranker.ORIGINAL and
            (run_config.stance_reranker_cutoff is None or
             run_config.stance_reranker_cutoff > 0)):
        name = run_config.stance_reranker.value
        if run_config.stance_reranker_cutoff is not None:
            name += f"@{run_config.stance_reranker_cutoff}"
        names.append(name)

    # Fair re-ranking.
    if run_config.fairness_reranker_cutoff is None:
        pipeline = pipeline >> run_config.fairness_reranker
    elif run_config.fairness_reranker_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = (
                           pipeline %
                           run_config.fairness_reranker_cutoff >>
                           run_config.fairness_reranker
                   ) ^ pipeline
    pipeline = ~pipeline

    if (run_config.fairness_reranker != StanceReranker.ORIGINAL and
            (run_config.fairness_reranker_cutoff is None or
             run_config.fairness_reranker_cutoff > 0)):
        name = run_config.fairness_reranker.value
        if (run_config.fairness_reranker ==
                FairnessReranker.BOOST_MINORITY_STANCE):
            name = "boost-min"
        if run_config.fairness_reranker_cutoff is not None:
            name += f"@{run_config.fairness_reranker_cutoff}"
        names.append(name)

    return NamedPipeline(names, pipeline)


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

    runs: list[NamedPipeline] = [
        _run(run_file_path, run_config)
        for team_directory_path in CONFIG.runs_directory_path.iterdir()
        if team_directory_path.is_dir()
        for run_file_path in (team_directory_path / "output").iterdir()
        for run_config in CONFIG.runs
    ]
    all_names: list[str] = [run.name for run in runs]
    all_systems: list[Transformer] = [run.pipeline for run in runs]

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
        column = column.replace("(groups='FIRST,SECOND,NEUTRAL')",
                                "(FIRST,SECOND,NEUTRAL)")
        return column

    experiment.columns = experiment.columns.map(rename_column)
    if CONFIG.metrics_output_file_path.suffix == ".csv":
        experiment.to_csv(CONFIG.metrics_output_file_path, index=False)
    if CONFIG.metrics_output_file_path.suffix == ".xlsx":
        if CONFIG.measures_per_query:
            logger.warning(
                "Evaluation measures per query "
                "might generate too large Excel sheet."
            )
        experiment.to_excel(CONFIG.metrics_output_file_path, index=False)
    if not CONFIG.measures_per_query:
        print(experiment.to_string(min_rows=len(experiment)))


if __name__ == '__main__':
    main()
