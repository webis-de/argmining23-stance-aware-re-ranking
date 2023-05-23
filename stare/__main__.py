from functools import cache
from math import sqrt, nan
from pathlib import Path
from textwrap import dedent
from typing import NamedTuple

from ir_measures import Measure
from pandas import DataFrame, merge, Series
from pyterrier import init
from pyterrier.io import read_qrels
from pyterrier.pipelines import Experiment
from pyterrier.transformer import Transformer
from scipy.stats import ttest_rel
from statsmodels.sandbox.stats.multicomp import MultiComparison

from stare.config import CONFIG, RunConfig
from stare.modules.diversity_reranker import DiversityReranker
from stare.modules.effectiveness_reranker import EffectivenessReranker
from stare.modules.fairness_reranker import FairnessReranker
from stare.modules.runs_loader import RunLoader
from stare.modules.stance_filter import StanceFilter
from stare.modules.stance_randomizer import StanceF1Randomizer
from stare.modules.stance_tagger import StanceTagger
from stare.modules.text_loader import TextLoader
from stare.modules.topics_loader import parse_topics

init(no_download=CONFIG.offline)


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


def _measure_cols(df: DataFrame) -> list[str]:
    effectiveness_relevance_suffix = (
        " rel." if len(CONFIG.measures_quality) > 0 else "")
    effectiveness_quality_suffix = (
        " qual." if len(CONFIG.measures_relevance) > 0 else "")
    diversity_relevance_suffix = (
        " rel." if len(CONFIG.measures_diversity_quality) > 0 else "")
    diversity_quality_suffix = (
        " qual." if len(CONFIG.measures_diversity_relevance) > 0 else "")
    columns = []
    for measure in CONFIG.measures_relevance:
        column = _rename_column(str(measure))
        columns.append(f"{column}{effectiveness_relevance_suffix}")
    for measure in CONFIG.measures_quality:
        column = _rename_column(str(measure))
        columns.append(f"{column}{effectiveness_quality_suffix}")
    for measure in CONFIG.measures_diversity_relevance:
        column = _rename_column(str(measure))
        columns.append(f"{column}{diversity_relevance_suffix}")
    for measure in CONFIG.measures_diversity_quality:
        column = _rename_column(str(measure))
        columns.append(f"{column}{diversity_quality_suffix}")
    for measure in CONFIG.measures_stance:
        column = _rename_column(str(measure))
        columns.append(column)
    return [
        column
        for column in columns
        if column in df.columns
    ]


def _compute_significance(df: DataFrame) -> DataFrame:
    significance_level = CONFIG.significance_level
    for measure_col in _measure_cols(df):
        comparison = MultiComparison(
            df[measure_col],
            df["name_index"],
        )
        _, _, results = comparison.allpairtest(
            ttest_rel,
            significance_level,
            "bonf",
        )
        lookup = {
            1: {
                "p-value": nan,
                "p-value_corrected": nan,
                "reject": False,
            }
        }
        for result in results:
            if result[0] == 1:
                name_index = result[1]
            elif result[1] == 1:
                name_index = result[0]
            else:
                continue
            lookup[name_index] = {
                "statistic": result[2],
                "p-value": result[3],
                "p-value_corrected": result[4],
                "reject": result[5],
            }
        df[f"{measure_col} p-value"] = [
            lookup[row["name_index"]]["p-value"]
            for _, row in df.iterrows()
        ]
        df[f"{measure_col} p-value corrected"] = [
            lookup[row["name_index"]]["p-value_corrected"]
            for _, row in df.iterrows()
        ]
        df[f"{measure_col} reject"] = [
            lookup[row["name_index"]]["reject"]
            for _, row in df.iterrows()
        ]

    return df


def cohen_d(x: Series, y: Series):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (
            (x.mean() - y.mean()) /
            sqrt(
                (
                        (nx - 1) * x.std(ddof=1) ** 2 +
                        (ny - 1) * y.std(ddof=1) ** 2
                ) / dof
            )
    )


def _compute_effect(df: DataFrame) -> DataFrame:
    for measure_col in _measure_cols(df):
        baseline = df[df["name_index"] == 1][measure_col]
        for name_index in df["name_index"].unique():
            compared = df[df["name_index"] == name_index][measure_col]
            # Cohen's d
            df.loc[
                df["name_index"] == name_index,
                f"{measure_col} effect"
            ] = cohen_d(compared, baseline)
    return df


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


def _rename_column(column: str) -> str:
    column = column.replace("group_col='stance_label'", "")
    column = column.replace("tie_breaking='group-ascending'", "")
    column = column.replace("tie_breaking='SECOND,FIRST,NEUTRAL,NO'", "")
    column = column.replace("groups='FIRST,NEUTRAL,SECOND'", "")
    column = column.replace(",,", ",")
    column = column.replace("(,", "(")
    column = column.replace(",)", ")")
    column = column.replace("()", "")
    return column


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

    experiment.columns = experiment.columns.map(_rename_column)

    def fix_name_order(df: DataFrame) -> DataFrame:
        df = df.set_index("name")
        df = df.loc[all_names]
        return df.reset_index(drop=False)

    experiment = experiment.groupby("qid").apply(fix_name_order)

    experiment["run"] = experiment["name"].apply(
        lambda name: name.split(" + ")[0])

    # Number names.
    experiment = experiment \
        .groupby(by="run", sort=False, group_keys=False) \
        .apply(_name_index) \
        .reset_index(drop=True)

    # Compute significance.
    if CONFIG.significance_level is not None:
        print("Compute significance.")
        experiment = experiment \
            .groupby(by="run", sort=False, group_keys=False) \
            .apply(_compute_significance)

    # Compute effect size.
    if CONFIG.effect_size:
        print("Compute effect size.")
        experiment = experiment \
            .groupby(by="run", sort=False, group_keys=False) \
            .apply(_compute_effect)
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
        measure_cols = _measure_cols(experiment)
        measure_names = [
            measure_col.replace("_", "\\_")
            for measure_col in measure_cols
        ]
        with open(output_path, "w") as file:
            file.write(dedent(r"""
            \newcommand{\significant}[1]{\ensuremath{\bm{#1}}}
            \newcommand{\effectup}[1]{\ensuremath{^{\uparrow#1}}}
            \newcommand{\effectdown}[1]{\ensuremath{^{\downarrow#1}}}
            \newcommand{\effectnone}[1]{\ensuremath{^{\phantom{\uparrow#1}}}}
            """).lstrip())
            file.write("\\begin{tabular}{l")
            file.write("c" * len(measure_cols))
            file.write("}\n")
            file.write("\\toprule\n")
            line = ["Ranking", *measure_names]
            file.write(" & ".join(line) + " \\\\\n")
            file.write("\\midrule\n")
            for _, row in experiment.iterrows():
                line = [
                    row["name"].replace("_", "\\_")
                ]
                for measure_col in measure_cols:
                    metric = row[measure_col]
                    column = f"{metric:.3f}"
                    significant = (row[f"{measure_col} p-value corrected"] <
                                   CONFIG.significance_level)
                    effect = row[f"{measure_col} effect"]
                    if effect > 0:
                        column = rf"{column}\effectup{{{effect:.1f}}}"
                    elif effect < 0:
                        column = rf"{column}\effectdown{{{-effect:.1f}}}"
                    else:
                        column = rf"{column}\effectnone{{0.0}}"
                    if significant:
                        column = rf"\significant{{{column}}}"
                    line.append(column)
                file.write(" & ".join(line) + " \\\\\n")
            file.write("\\bottomrule\n")
            file.write("\\end{tabular}\n")


if __name__ == '__main__':
    main()
