from typing import Tuple, Iterable, Callable

from ir_measures import define
from pandas import DataFrame
from sklearn.metrics import f1_score


def _f1_score(
        qrels: DataFrame,
        run: DataFrame,
) -> Iterable[Tuple[str, float]]:
    df = qrels.merge(
        run,
        on=["query_id", "doc_id"],
        how="left",
        suffixes=("_qrels", "_run"),
    )
    df = df.dropna(subset=["stance_label_qrels", "stance_label_run"])
    score = f1_score(
        df["stance_label_qrels"],
        df["stance_label_run"],
        average="macro",
    )
    for qid in df["query_id"].unique():
        yield qid, score


F1 = define(
    _f1_score,
    name="F1",
    support_cutoff=True,
)


def _judged_stance(
        qrels: DataFrame,
        run: DataFrame,
) -> Iterable[Tuple[str, float]]:
    df = qrels.merge(
        run,
        on=["query_id", "doc_id"],
        how="inner",
        suffixes=("_qrels", "_run"),
    )
    judged = len(df)
    for qid in df["query_id"].unique():
        yield qid, judged


NumJudged = define(
    _judged_stance,
    name="NumJudged",
    support_cutoff=True,
)


def _frequency(
        label: str
) -> Callable[[DataFrame, DataFrame], Iterable[Tuple[str, float]]]:
    def _wrapped(
            _: DataFrame,
            run: DataFrame,
    ) -> Iterable[Tuple[str, float]]:
        frequency = len(run[run["stance_label"] == label]) / len(run)
        for qid in run["query_id"].unique():
            yield qid, frequency

    return _wrapped


FreqFirst = define(
    _frequency("FIRST"),
    name="FreqFirst",
    support_cutoff=True,
)
FreqSecond = define(
    _frequency("SECOND"),
    name="FreqSecond",
    support_cutoff=True,
)
FreqNeutral = define(
    _frequency("NEUTRAL"),
    name="FreqNeutral",
    support_cutoff=True,
)
FreqNo = define(
    _frequency("NO"),
    name="FreqNo",
    support_cutoff=True,
)
