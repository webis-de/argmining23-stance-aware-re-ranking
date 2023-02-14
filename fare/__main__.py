from functools import cache

from ir_measures import Measure
from pyterrier import init

from fare.config import CONFIG
from fare.modules.diversity_reranker import DiversityReranker

init(no_download=CONFIG.offline)

from pathlib import Path
from typing import NamedTuple

from pandas import DataFrame, merge, Series
from pyterrier.io import read_qrels
from pyterrier.pipelines import Experiment
from pyterrier.transformer import Transformer
from scipy.stats import ttest_rel
from statsmodels.sandbox.stats.multicomp import MultiComparison

from fare.config import RunConfig
from fare.modules.fairness_reranker import FairnessReranker
from fare.modules.runs_loader import RunLoader
from fare.modules.stance_filter import StanceFilter
from fare.modules.stance_randomizer import StanceF1Randomizer
from fare.modules.effectiveness_reranker import EffectivenessReranker
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
    pipeline = ~(
            pipeline >>
            TextLoader()
    )

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
        pipeline = (
                ~(
                        ~(
                                pipeline %
                                run_config.stance_tagger_cutoff >>
                                run_config.stance_tagger
                        ) >>
                        StanceFilter(run_config.stance_tagger_threshold)
                ) ^
                pipeline
        )

    if run_config.stance_tagger != StanceTagger.ORIGINAL:
        name = run_config.stance_tagger.value
        if run_config.stance_tagger_threshold > 0:
            name += f"({run_config.stance_tagger_threshold:.2f})"
        if run_config.stance_tagger_cutoff is not None:
            name += f"@{run_config.stance_tagger_cutoff}"
        names.append(name)

    # Randomize stance.
    if run_config.stance_randomization_cutoff is None:
        pipeline = ~(
                pipeline >>
                StanceF1Randomizer(run_config.stance_randomization_target_f1)
        )
    elif run_config.stance_randomization_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = (
                ~(
                        pipeline %
                        run_config.stance_randomization_cutoff >>
                        StanceF1Randomizer(
                            run_config.stance_randomization_target_f1)
                ) ^
                pipeline
        )

    if run_config.stance_randomization_target_f1 < 1:
        name = "randomize"
        name += f"(F1<={run_config.stance_randomization_target_f1:.2f})"
        if run_config.stance_randomization_cutoff is not None:
            name += f"@{run_config.stance_randomization_cutoff}"
        names.append(name)

    # Re-rank for effectiveness.
    if run_config.effectiveness_reranker_cutoff is None:
        pipeline = ~(
                pipeline >>
                run_config.effectiveness_reranker
        )
    elif run_config.effectiveness_reranker_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = (
                ~(
                        pipeline %
                        run_config.effectiveness_reranker_cutoff >>
                        run_config.effectiveness_reranker
                ) ^
                pipeline
        )

    if (run_config.effectiveness_reranker != EffectivenessReranker.ORIGINAL and
            (run_config.effectiveness_reranker_cutoff is None or
             run_config.effectiveness_reranker_cutoff > 0)):
        name = run_config.effectiveness_reranker.value
        if run_config.effectiveness_reranker_cutoff is not None:
            name += f"@{run_config.effectiveness_reranker_cutoff}"
        names.append(name)

    # Re-rank for diversity.
    if run_config.diversity_reranker_cutoff is None:
        pipeline = ~(
                pipeline >>
                run_config.diversity_reranker
        )
    elif run_config.diversity_reranker_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = (
                ~(
                        pipeline %
                        run_config.diversity_reranker_cutoff >>
                        run_config.diversity_reranker
                ) ^
                pipeline
        )

    if (run_config.diversity_reranker != DiversityReranker.ORIGINAL and
            (run_config.diversity_reranker_cutoff is None or
             run_config.diversity_reranker_cutoff > 0)):
        name = run_config.diversity_reranker.value
        if run_config.diversity_reranker_cutoff is not None:
            name += f"@{run_config.diversity_reranker_cutoff}"
        names.append(name)

    # Re-rank for fairness.
    if run_config.fairness_reranker_cutoff is None:
        pipeline = ~(
                pipeline >>
                run_config.fairness_reranker
        )
    elif run_config.fairness_reranker_cutoff > 0:
        # noinspection PyTypeChecker
        pipeline = (
                ~(
                        pipeline %
                        run_config.fairness_reranker_cutoff >>
                        run_config.fairness_reranker
                ) ^
                pipeline
        )

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


def _name_index(experiment: DataFrame) -> DataFrame:
    names = experiment["name"].unique()
    name_index = {name: i + 1 for i, name in enumerate(names)}
    columns = experiment.columns.tolist()
    experiment["name_index"] = experiment["name"].map(name_index)
    columns.insert(1, "name_index")
    columns.remove("index")
    return experiment[columns]


def _significance_test(experiment: DataFrame) -> DataFrame:
    significance_level = CONFIG.significance_level
    if significance_level is None:
        return experiment

    measure_columns = [
        measure
        for measure in experiment.columns
        if measure not in ("qid", "run", "name", "name_index", "index")
    ]

    for measure in measure_columns:
        comparison = MultiComparison(
            experiment[measure],
            experiment["name_index"],
        )
        _, _, results = comparison.allpairtest(
            ttest_rel,
            significance_level,
            "bonf",
        )
        results = DataFrame([
            {
                "index1": result[0],
                "index2": result[1],
                "statistic": result[2],
                "pvalue": result[3],
                "pvalue_corrected": result[4],
                "reject": result[5],
            }
            for result in results
        ])

        def significant_to(index: str) -> str:
            df = results
            df = df[(df["index1"] == index) | (df["index2"] == index)]
            df = df.loc[results["reject"]]
            significant = {*df["index1"], *df["index2"]} - {index}
            indices = sorted(significant)
            return ",".join(map(str, indices))

        experiment[f"{measure} test"] = \
            experiment["name_index"].map(significant_to)

    new_columns = ["qid", "run", "name", "name_index"]
    for measure in measure_columns:
        new_columns.append(measure)
        new_columns.append(f"{measure} test")
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
    qrels_diversity_relevance = _diversity_qrels(
        qrels_relevance, qrels_stance
    )
    qrels_diversity_quality = _diversity_qrels(
        qrels_relevance, qrels_stance
    )

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
    if CONFIG.team_runs is not None:
        runs = [
            run
            for run in runs
            if any(
                run.name.startswith(team_run)
                for team_run in CONFIG.team_runs
            )
        ]
    all_names: list[str] = [run.name for run in runs]

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
        suffixes=(" rel.", " qual.")
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
        suffixes=(" rel.", " qual.")
    )

    print("Compute stance measures.")
    # noinspection PyTypeChecker
    stance = _run_experiment(
        runs,
        topics,
        qrels_stance,
        CONFIG.measures_stance,
    )

    experiment = effectiveness.merge(
        diversity,
        on=["qid", "name"],
        suffixes=("", " stance")
    ).merge(
        stance,
        on=["qid", "name"],
        suffixes=("", " diversity")
    ).reset_index(drop=False)


    def rename_column(column: str) -> str:
        column = column.replace("group_col='stance_label'", "")
        column = column.replace("tie_breaking='group-ascending'", "")
        column = column.replace("tie_breaking='SECOND,FIRST,NEUTRAL,NO'", "")
        column = column.replace("groups='FIRST,NEUTRAL,SECOND'", "")
        column = column.replace(",,", ",")
        column = column.replace("(,", "(")
        column = column.replace(",)", ")")
        column = column.replace("()", "")
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
        .apply(_name_index).reset_index(drop=True)
    if CONFIG.significance_level is not None:
        print("Compute significance.")
        experiment = experiment.groupby(by="run", sort=False, group_keys=False) \
            .apply(_significance_test)
    del experiment["run"]

    # Aggregate results.
    if not CONFIG.measures_per_query:
        print("Aggregate results.")
        aggregations = {
            column:
                "first"
                if (column.endswith(" test") or column == "run"
                    or column == "name" or column == "name_index")
                else "mean"
            for column in experiment.columns
            if column != "qid"
        }
        experiment = experiment \
            .groupby(by=["name", "name_index"], sort=False,
                     group_keys=True) \
            .aggregate(aggregations)

    # Export results.
    print("Export results.")
    output_path = CONFIG.metrics_output_file_path
    if CONFIG.measures_per_query:
        output_path = output_path.with_suffix(
            f".perquery{output_path.suffix}"
        )
    if output_path.suffix == ".csv":
        experiment.to_csv(output_path, index=False, float_format="%.3f")
    if output_path.suffix == ".xlsx":
        experiment.to_excel(output_path, index=False)
    if output_path.suffix == ".tex":
        measures_suffixes = [
            (
                CONFIG.measures_relevance,
                " rel." if len(CONFIG.measures_quality) > 0 else "",
            ),
            (
                CONFIG.measures_quality,
                " qual." if len(CONFIG.measures_relevance) > 0 else "",
            ),
            (
                CONFIG.measures_diversity_relevance,
                " rel." if len(CONFIG.measures_diversity_quality) > 0 else "",
            ),
            (
                CONFIG.measures_diversity_quality,
                " qual." if len(
                    CONFIG.measures_diversity_relevance) > 0 else "",
            ),
            (
                CONFIG.measures_stance,
                "",
            ),
        ]
        measure_names = [
            str(measure).replace("_", "\\_") + suffix
            for measures, suffix in measures_suffixes
            for measure in measures
        ]
        measure_cols = [
            rename_column(f"{measure}{suffix}")
            for measures, suffix in measures_suffixes
            for measure in measures
        ]
        with open(output_path, "w") as file:
            file.write(
                "\\begin{tabular}{" + "l" * (
                        len(measure_names) + 2) + "}\n")
            file.write("\\toprule\n")
            line = ["Run", "\\#", *measure_names]
            file.write(" & ".join(line) + " \\\\\n")
            file.write("\\midrule\n")
            for _, row in experiment.iterrows():
                line = [
                    row["name"].replace("_", "\\_"),
                    str(row["name_index"]),
                ]
                for measure_col in measure_cols:
                    metric = row[measure_col]
                    if CONFIG.significance_level is not None:
                        significant = row[f"{measure_col} test"]
                        line.append(
                            f"{metric:.3f}\\(^{{{significant}}}\\)")
                    else:
                        line.append(f"{metric:.3f}")
                file.write(" & ".join(line) + " \\\\\n")
            file.write("\\bottomrule\n")
            file.write("\\end{tabular}\n")


if __name__ == '__main__':
    main()
