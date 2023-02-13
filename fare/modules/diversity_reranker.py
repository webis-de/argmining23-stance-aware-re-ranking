from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from math import nan
from typing import List

from numpy import arange
from pandas import DataFrame, Series
from pyterrier import Transformer
from pyterrier.model import add_ranks


@dataclass(frozen=True)
class StanceStreakDiversityReranker(Transformer):

    @staticmethod
    def _transform_query(ranking: DataFrame) -> DataFrame:
        new_rows: List[Series] = []
        last_stance: float = nan
        while len(ranking) > 0:
            candidates: DataFrame
            if last_stance > 0:
                # Last document was pro A.
                # Find first pro B or neutral document next.
                candidates = ranking[ranking["stance_value"] <= 0]
            elif last_stance < 0:
                # Last document was pro B.
                # Find first pro A or neutral document next.
                candidates = ranking[ranking["stance_value"] >= 0]
            else:
                # Last document was neutral.
                # Find any document next, regardless of stance.
                candidates = ranking

            if len(candidates) == 0:
                # No candidate for the stance was found,
                # choose any document next,
                # regardless of stance.
                last_stance = nan
                continue

            index = candidates.index.tolist()[0]
            document: Series = candidates.iloc[0]
            last_stance: float = document["stance_value"]
            new_rows.append(document)
            ranking.drop(index=index, inplace=True)
        ranking = DataFrame(data=new_rows, columns=ranking.columns)

        # Reset score.
        ranking["score"] = arange(len(ranking), 0, -1)
        return ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        groups = ranking.groupby("qid", sort=False, group_keys=False)
        ranking = groups.apply(self._transform_query)
        ranking = ranking.reset_index(drop=True)
        ranking = add_ranks(ranking)
        return ranking


@dataclass(frozen=True)
class AlternatingStanceDiversityReranker(Transformer):
    revision: int = 2

    @staticmethod
    def _transform_query(ranking: DataFrame) -> DataFrame:
        new_rows: List[Series] = []
        last_stance: float = nan
        while len(ranking) > 0:
            candidates: DataFrame
            if last_stance > 0:
                # Last document was pro A.
                # Find first pro B or neutral document next.
                candidates = ranking[ranking["stance_value"] <= 0]
            elif last_stance < 0:
                # Last document was pro B.
                # Find first pro A or neutral document next.
                candidates = ranking[ranking["stance_value"] >= 0]
            else:
                # Last document was neutral.
                # Find any document next, regardless of stance.
                candidates = ranking

            if len(candidates) == 0:
                # No candidate for the stance was found,
                # choose any document next,
                # regardless of stance.
                last_stance = nan
                continue

            index = candidates.index.tolist()[0]
            document: Series = candidates.iloc[0]
            if document["stance_value"] > 0 or document["stance_value"] < 0:
                last_stance = document["stance_value"]
            new_rows.append(document)
            ranking.drop(index=index, inplace=True)
        ranking = DataFrame(data=new_rows, columns=ranking.columns)

        # Reset score.
        ranking["score"] = arange(len(ranking), 0, -1)
        return ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        groups = ranking.groupby("qid", sort=False, group_keys=False)
        ranking = groups.apply(self._transform_query)
        ranking = ranking.reset_index(drop=True)
        ranking = add_ranks(ranking)
        return ranking


class DiversityReranker(Transformer, Enum):
    ORIGINAL = "original"
    STANCE_STREAK = "stance-streak"
    ALTERNATING_STANCE = "alternating-stance"

    @cached_property
    def _transformer(self) -> Transformer:
        if self == DiversityReranker.ORIGINAL:
            return Transformer.identity()
        elif self == DiversityReranker.STANCE_STREAK:
            return StanceStreakDiversityReranker()
        elif self == DiversityReranker.ALTERNATING_STANCE:
            return AlternatingStanceDiversityReranker()
        else:
            raise ValueError(f"Unknown diversity re-ranker: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self._transformer.transform(ranking)

    def __repr__(self) -> str:
        return repr(self._transformer)
