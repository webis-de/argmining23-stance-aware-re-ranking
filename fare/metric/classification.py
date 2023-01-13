from math import nan
from typing import Tuple, Iterable, Callable

from ir_measures import define
from pandas import DataFrame, merge
from sklearn.metrics import f1_score


def _f1_score(
        qrels: DataFrame,
        run: DataFrame,
        across_all_queries: bool = False
) -> Iterable[Tuple[str, float]]:
    df = merge(
        qrels,
        run,
        on=["query_id", "doc_id"],
        how="left",
        suffixes=("_qrels", "_run"),
    )
    df = df.dropna(subset=["stance_label_qrels", "stance_label_run"])
    if across_all_queries:
        score = f1_score(
            df["stance_label_qrels"],
            df["stance_label_run"],
            average="macro",
        )
        for qid in df["query_id"].unique():
            yield qid, score
    else:
        for qid, df in df.groupby("query_id", sort=False):
            yield qid, f1_score(
                df["stance_label_qrels"],
                df["stance_label_run"],
                average="macro",
            )


def _f1_score_touche(
        qrels: DataFrame,
        run: DataFrame,
) -> Iterable[Tuple[str, float]]:
    return _f1_score(qrels, run, across_all_queries=True)


F1 = define(_f1_score, name="F1", support_cutoff=True)

F1_Touche = define(_f1_score_touche, name="F1_Touche", support_cutoff=True)


def _confidence(
        _: DataFrame,
        run: DataFrame,
) -> Iterable[Tuple[str, float]]:
    for qid, df in run.groupby("query_id", sort=False):
        stance = df["stance_value"] \
            .replace(0, nan) \
            .dropna()\
            .abs()
        yield qid, stance.mean()


Confidence = define(_confidence, name="Confidence", support_cutoff=True)


def _proportion(
        label: str
) -> Callable[[DataFrame, DataFrame], Iterable[Tuple[str, float]]]:
    def _wrapped(
            _: DataFrame,
            run: DataFrame,
    ) -> Iterable[Tuple[str, float]]:
        for qid, df in run.groupby("query_id", sort=False):
            df = df.dropna(subset=["stance_label"])
            proportion = len(df[df["stance_label"] == label]) / len(df)
            yield qid, proportion

    return _wrapped


PropFirst = define(_proportion("FIRST"), name="PropFirst", support_cutoff=True)
PropSecond = define(_proportion("SECOND"), name="PropSecond",
                    support_cutoff=True)
PropNeutral = define(_proportion("NEUTRAL"), name="PropNeutral",
                     support_cutoff=True)
PropNo = define(_proportion("NO"), name="PropNo", support_cutoff=True)
