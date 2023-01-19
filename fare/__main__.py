from functools import cache

from ir_measures import Measure
from pyterrier import init

from fare.config import CONFIG

init(no_download=CONFIG.offline)

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

from pandas import DataFrame, merge, Series
from pyterrier.io import read_qrels
from pyterrier.pipelines import Experiment
from pyterrier.transformer import Transformer
from scipy.stats import ttest_rel

from fare.config import RunConfig
from fare.modules.fairness_reranker import FairnessReranker
from fare.modules.runs_loader import RunLoader
from fare.modules.stance_filter import StanceFilter
from fare.modules.stance_reranker import StanceReranker
from fare.modules.stance_tagger import StanceTagger
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
    names = [f"{team_directory_path.stem} {pipeline.name}"]

    # Load text contents.
    pipeline = pipeline >> TextLoader()
    pipeline = ~pipeline

    # Tag stance.
    if run_config.stance_tagger_cutoff is None:
        pipeline = ~(
                ~(
                        pipeline >>
                        run_config.stance_tagger
                ) >>
                StanceFilter(run_config.stance_tagger_threshold)
        )
    elif run_config.stance_tagger_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = ~(
                ~(
                        pipeline %
                        run_config.stance_tagger_cutoff >>
                        run_config.stance_tagger
                ) >> StanceFilter(run_config.stance_tagger_threshold)
        ) ^ pipeline

    if run_config.stance_tagger != StanceTagger.ORIGINAL:
        name = run_config.stance_tagger.value
        if run_config.stance_tagger_threshold > 0:
            name += f"({run_config.stance_tagger_threshold:f})"
        if run_config.stance_tagger_cutoff is not None:
            name += f"@{run_config.stance_tagger_cutoff}"
        names.append(name)

    # Re-rank stance/subjective first.
    if run_config.stance_reranker_cutoff is None:
        pipeline = ~(
                pipeline >>
                run_config.stance_reranker
        )
    elif run_config.stance_reranker_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = ~(
                pipeline %
                run_config.stance_reranker_cutoff >>
                run_config.stance_reranker
        ) ^ pipeline

    if (run_config.stance_reranker != StanceReranker.ORIGINAL and
            (run_config.stance_reranker_cutoff is None or
             run_config.stance_reranker_cutoff > 0)):
        name = run_config.stance_reranker.value
        if run_config.stance_reranker_cutoff is not None:
            name += f"@{run_config.stance_reranker_cutoff}"
        names.append(name)

    # Fair re-ranking.
    if run_config.fairness_reranker_cutoff is None:
        pipeline = ~(
                pipeline >>
                run_config.fairness_reranker
        )
    elif run_config.fairness_reranker_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = ~(
                pipeline %
                run_config.fairness_reranker_cutoff >>
                run_config.fairness_reranker
        ) ^ pipeline

    if (run_config.fairness_reranker != FairnessReranker.ORIGINAL and
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


def is_significant(
        experiment: DataFrame,
        significance_level: float,
        name1: str,
        name2: str,
        measure: str,
) -> bool:
    experiment1 = experiment[experiment["name"] == name1]
    experiment2 = experiment[experiment["name"] == name2]
    t_test = ttest_rel(experiment1[measure], experiment2[measure])
    significant = t_test.pvalue < significance_level
    return significant


def _pairwise_t_test(experiment: DataFrame) -> DataFrame:
    experiment = experiment.copy()

    names = experiment["name"].unique()
    name_index = {name: i + 1 for i, name in enumerate(names)}
    name_pairs = list(combinations(names, 2))

    experiment["name_index"] = experiment["name"].map(name_index)

    measure_columns = [
        measure
        for measure in experiment.columns
        if measure not in ("qid", "run", "name", "name_index", "index")
    ]

    significance_level = CONFIG.significance_level
    if significance_level is not None:
        if len(name_pairs) > 0:
            # Bonferroni correction.
            significance_level /= len(name_pairs)

        for measure in measure_columns:
            pairwise_significance: dict[str, set[int]] = defaultdict(set)
            for run1, run2 in name_pairs:
                if is_significant(
                        experiment,
                        significance_level,
                        run1, run2,
                        measure,
                ):
                    pairwise_significance[run1].add(name_index[run2])
                    pairwise_significance[run2].add(name_index[run1])

            experiment[f"{measure} t-test"] = experiment["name"].map(
                lambda name: ",".join(
                    map(str, sorted(pairwise_significance[name]))
                )
            )

    new_columns = ["qid", "run", "name", "name_index"]
    for measure in measure_columns:
        new_columns.append(measure)
        if significance_level is not None:
            new_columns.append(f"{measure} t-test")
    return experiment[new_columns]


@cache
def _diversity_label(label: float, stance: str, subtopic: str) -> float:
    if subtopic == stance:
        return label
    if subtopic in ("FIRST", "SECOND") and stance == "NEUTRAL":
        return label / 2
    if subtopic == "NEUTRAL" and stance in ("FIRST", "SECOND"):
        return label / 2
    else:
        return 0


def _diversity_qrels(
        effectiveness_qrels: DataFrame,
        stance_qrels: DataFrame,
) -> DataFrame:
    qrels = effectiveness_qrels.merge(
        stance_qrels,
        on=["qid", "docno"],
        how="inner",
        suffixes=(None, "_stance"),
    )
    return DataFrame([
        {
            "qid": row["qid"],
            "docno": row["docno"],
            "iteration": subtopic,
            "label": _diversity_label(
                row["label"],
                row["label_stance"],
                subtopic,
            ),
        }
        for _, row in qrels.iterrows()
        for subtopic in {"FIRST", "SECOND", "NEUTRAL"}
    ])


def _run_experiment(
        runs: list[NamedPipeline],
        topics: DataFrame,
        qrels: DataFrame,
        measures: list[Measure],
) -> DataFrame:
    all_names: list[str] = [run.name for run in runs]
    all_systems: list[Transformer] = [run.pipeline for run in runs]
    if len(measures) == 0:
        return merge(
            topics["qid"],
            Series(all_names, name="name"),
            how="cross",
        )
    return Experiment(
        retr_systems=all_systems,
        topics=topics,
        qrels=qrels,
        eval_metrics=measures,
        names=all_names,
        filter_by_qrels=CONFIG.filter_by_qrels,
        verbose=True,
        perquery=True,
    ).pivot_table(
        index=["qid", "name"],
        columns="measure",
        values="value",
        aggfunc="first",
    ).reset_index(
        drop=False
    )


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
    qrels_diversity_relevance = _diversity_qrels(qrels_relevance, qrels_stance)
    qrels_diversity_quality = _diversity_qrels(qrels_relevance, qrels_stance)

    max_teams = CONFIG.max_teams \
        if CONFIG.max_teams is not None else None
    max_runs_per_team = CONFIG.max_runs_per_team \
        if CONFIG.max_runs_per_team is not None else None
    runs: list[NamedPipeline] = [
        _run(run_file_path, run_config)
        for team_directory_path in
        sorted(
            CONFIG.runs_directory_path.iterdir()
        )[:max_teams]
        if team_directory_path.is_dir()
        for run_file_path in
        sorted(
            (team_directory_path / "output").iterdir()
        )[:max_runs_per_team]
        for run_config in CONFIG.runs
    ]
    all_names: list[str] = [run.name for run in runs]
    all_systems: list[Transformer] = [run.pipeline for run in runs]


    print("Compute relevance effectiveness measures.")
    effectiveness_relevance = _run_experiment(
        runs,
        topics,
        qrels_relevance,
        CONFIG.measures_relevance,
    )
    print("Compute quality effectiveness measures.")
    effectiveness_quality = _run_experiment(
        runs,
        topics,
        qrels_quality,
        CONFIG.measures_quality,
    )
    effectiveness = effectiveness_relevance.merge(
        effectiveness_quality,
        on=["qid", "name"],
        suffixes=(" relevance", " quality")
    )

    print("Compute stance measures.")
    # noinspection PyTypeChecker
    stance = _run_experiment(
        runs,
        topics,
        qrels_stance,
        CONFIG.measures_stance,
    )

    print("Compute relevance diversity measures.")
    diversity_relevance = _run_experiment(
        runs,
        topics,
        qrels_diversity_relevance,
        CONFIG.measures_diversity_relevance,
    )
    print("Compute quality diversity measures.")
    diversity_quality = _run_experiment(
        runs,
        topics,
        qrels_diversity_quality,
        CONFIG.measures_diversity_quality,
    )
    diversity = diversity_relevance.merge(
        diversity_quality,
        on=["qid", "name"],
        suffixes=(" relevance", " quality")
    )

    experiment = effectiveness.merge(
        stance,
        on=["qid", "name"],
        suffixes=("", " stance")
    ).merge(
        diversity,
        on=["qid", "name"],
        suffixes=("", " diversity")
    ).reset_index(drop=False)

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

    def fix_name_order(df: DataFrame) -> DataFrame:
        df = df.set_index("name")
        df = df.loc[all_names]
        return df.reset_index(drop=False)

    experiment = experiment.groupby("qid").apply(fix_name_order)

    experiment["run"] = experiment["name"].apply(
        lambda name: name.split(" + ")[0]
    )
    experiment = experiment.groupby(by="run", sort=False, group_keys=False) \
        .apply(_pairwise_t_test)
    del experiment["run"]

    # Aggregate results.
    if not CONFIG.measures_per_query:
        aggregations = {
            column:
                "first"
                if (column.endswith("t-test") or column == "run"
                    or column == "name" or column == "name_index")
                else "mean"
            for column in experiment.columns
            if column != "qid"
        }
        experiment = experiment \
            .groupby(by=["name", "name_index"], sort=False, group_keys=True) \
            .aggregate(aggregations)

        # Export results.
    output_path = CONFIG.metrics_output_file_path
    if CONFIG.measures_per_query:
        output_path = output_path.with_suffix(
            f".perquery{output_path.suffix}"
        )
    if output_path.suffix == ".csv":
        experiment.to_csv(output_path, index=False, float_format="%.3f")
    if output_path.suffix == ".xlsx":
        experiment.to_excel(output_path, index=False)


if __name__ == '__main__':
    main()
