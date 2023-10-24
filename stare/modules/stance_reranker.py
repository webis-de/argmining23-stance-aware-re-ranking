from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from pandas import DataFrame
from pyterrier.model import add_ranks
from pyterrier.transformer import Transformer
from tqdm.auto import tqdm


@dataclass(frozen=True)
class StanceFirstReranker(Transformer):
    verbose: bool = False

    @staticmethod
    def _rerank_query(ranking: DataFrame) -> DataFrame:
        ranking["has_stance"] = ranking["stance_label"].isin({
            "FIRST",
            "SECOND",
            "NEUTRAL"
        })
        ranking.sort_values(
            "has_stance",
            ascending=False,
            inplace=True
        )
        del ranking["has_stance"]

        # Reset rank and score.
        ranking["score"] = list(range(len(ranking), 0, -1))

        return ranking

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        groups = topics_or_res.groupby("qid", sort=False, group_keys=False)
        if self.verbose:
            tqdm.pandas(desc="Rank stance first", unit="query")
            groups = groups.progress_apply(self._rerank_query)
        else:
            groups = groups.apply(self._rerank_query)
        topics_or_res = groups.reset_index(drop=True)
        topics_or_res = add_ranks(topics_or_res)
        return topics_or_res


@dataclass(frozen=True)
class SubjectiveStanceFirstReranker(Transformer):
    verbose: bool = False

    @staticmethod
    def _rerank_query(ranking: DataFrame) -> DataFrame:
        ranking["has_stance"] = ranking["stance_label"].isin({
            "FIRST",
            "SECOND",
            "NEUTRAL"
        })
        ranking["is_subjective"] = ranking["stance_label"].isin({
            "FIRST",
            "SECOND",
        })
        ranking.sort_values(
            ["has_stance", "is_subjective"],
            ascending=False,
            inplace=True
        )
        del ranking["has_stance"]
        del ranking["is_subjective"]

        # Reset rank and score.
        ranking["score"] = list(range(len(ranking), 0, -1))

        return ranking

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        groups = topics_or_res.groupby("qid", sort=False, group_keys=False)
        if self.verbose:
            tqdm.pandas(desc="Rank subjective stance first", unit="query")
            groups = groups.progress_apply(self._rerank_query)
        else:
            groups = groups.apply(self._rerank_query)
        topics_or_res = groups.reset_index(drop=True)
        topics_or_res = add_ranks(topics_or_res)
        return topics_or_res


class StanceReranker(Transformer, Enum):
    ORIGINAL = "original"
    STANCE_FIRST = "stance-first"
    SUBJECTIVE_STANCE_FIRST = "subjective-stance-first"

    @cached_property
    def _transformer(self) -> Transformer:
        if self == StanceReranker.ORIGINAL:
            return Transformer.identity()
        elif self == StanceReranker.STANCE_FIRST:
            return StanceFirstReranker()
        elif self == StanceReranker.SUBJECTIVE_STANCE_FIRST:
            return SubjectiveStanceFirstReranker()
        else:
            raise ValueError(f"Unknown effectiveness re-ranker: {self}")

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return self._transformer.transform(topics_or_res)

    def __repr__(self) -> str:
        return repr(self._transformer)
