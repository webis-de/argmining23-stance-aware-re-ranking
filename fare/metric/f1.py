from typing import Tuple, Iterable

from ir_measures import define
from pandas import DataFrame, merge
from sklearn.metrics import f1_score


def _f1_score(
        qrels: DataFrame,
        run: DataFrame,
) -> Iterable[Tuple[str, float]]:
    df = merge(
        qrels,
        run,
        on=["query_id", "doc_id"],
        how="right",
        suffixes=("_qrels", "_run"),
    )
    df = df[~df["stance_label_qrels"].isna() & ~df["stance_label_run"].isna()]
    for qid, df in df.groupby("query_id", sort=False):
        yield qid, f1_score(
            df["stance_label_qrels"],
            df["stance_label_run"],
            average="macro",
        )


F1 = define(_f1_score, name="F1")
