from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from pandas import DataFrame
from pyterrier.transformer import Transformer, IdentityTransformer
from tqdm.auto import tqdm


@dataclass(frozen=True)
class StanceFirstReranker(Transformer):
    verbose: bool = False

    @staticmethod
    def _rerank_query(ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
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
        ranking["rank"] = list(range(1, len(ranking) + 1))
        ranking["score"] = list(range(len(ranking), 0, -1))

        return ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        groups = ranking.groupby("qid", sort=False, group_keys=False)
        if self.verbose:
            tqdm.pandas(desc="Rank stance first", unit="query")
            groups = groups.progress_apply(self._rerank_query)
        else:
            groups = groups.apply(self._rerank_query)
        return groups.reset_index(drop=True)


@dataclass(frozen=True)
class SubjectiveStanceFirstReranker(Transformer):
    verbose: bool = False

    @staticmethod
    def _rerank_query(ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
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
        ranking["rank"] = list(range(1, len(ranking) + 1))
        ranking["score"] = list(range(len(ranking), 0, -1))

        return ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        groups = ranking.groupby("qid", sort=False, group_keys=False)
        if self.verbose:
            tqdm.pandas(desc="Rank subjective stance first", unit="query")
            groups = groups.progress_apply(self._rerank_query)
        else:
            groups = groups.apply(self._rerank_query)
        return groups.reset_index(drop=True)


class StanceReranker(Transformer, Enum):
    ORIGINAL = "original"
    STANCE_FIRST = "stance-first"
    SUBJECTIVE_STANCE_FIRST = "subjective-stance-first"

    @cached_property
    def _transformer(self) -> Transformer:
        if self == StanceReranker.ORIGINAL:
            return IdentityTransformer()
        elif self == StanceReranker.STANCE_FIRST:
            return StanceFirstReranker(verbose=True)
        elif self == StanceReranker.SUBJECTIVE_STANCE_FIRST:
            return SubjectiveStanceFirstReranker(verbose=True)
        else:
            raise ValueError(f"Unknown stance re-ranker: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self._transformer.transform(ranking)
