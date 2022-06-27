from enum import Enum
from functools import cached_property

from pandas import DataFrame
from pyterrier.transformer import Transformer, IdentityTransformer


class StanceFirstReranker(Transformer):

    @staticmethod
    def rerank_query(ranking: DataFrame) -> DataFrame:
        ranking["has_stance"] = ranking["stance_label"].isin([
            "FIRST",
            "SECOND",
            "NEUTRAL"
        ])
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
        group_by_query = ranking.groupby("qid")
        return group_by_query.apply(self.rerank_query).reset_index(drop=True)


class SubjectiveStanceFirstReranker(Transformer):

    @staticmethod
    def rerank_query(ranking: DataFrame) -> DataFrame:
        ranking["has_stance"] = ranking["stance_label"].isin([
            "FIRST",
            "SECOND",
            "NEUTRAL"
        ])
        ranking["is_subjective"] = ranking["stance_label"].isin([
            "FIRST",
            "SECOND",
        ])
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
        group_by_query = ranking.groupby("qid")
        return group_by_query.apply(self.rerank_query).reset_index(drop=True)


class StanceReranker(Transformer, Enum):
    ORIGINAL = "original"
    STANCE_FIRST = "stance-first"
    SUBJECTIVE_STANCE_FIRST = "subjective-stance-first"

    @cached_property
    def transformer(self) -> Transformer:
        if self == StanceReranker.ORIGINAL:
            return IdentityTransformer()
        elif self == StanceReranker.STANCE_FIRST:
            return StanceFirstReranker()
        elif self == StanceReranker.SUBJECTIVE_STANCE_FIRST:
            return SubjectiveStanceFirstReranker()
        else:
            raise ValueError(f"Unknown stance re-ranker: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self.transformer.transform(ranking)
