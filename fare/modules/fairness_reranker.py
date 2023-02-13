from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Sequence

from pandas import DataFrame, Series
from pyterrier.model import add_ranks
from pyterrier.transformer import Transformer
from tqdm.auto import tqdm


def _normalize_scores(ranking: DataFrame, inplace: bool = False) -> DataFrame:
    ranking = ranking.copy()
    min_score = ranking["score"].min()
    max_score = ranking["score"].max()
    if not inplace:
        ranking = ranking.copy()
    ranking["score"] = (
            (ranking["score"] - min_score) /
            (max_score - min_score)
    )
    return ranking


@dataclass(frozen=True)
class InverseStanceGainFairnessReranker(Transformer):
    stances: Sequence[str] = ("FIRST", "SECOND", "NEUTRAL", "NO")
    alpha: float = 0.5
    verbose: bool = False

    @staticmethod
    def _discounted_gain(ranking_stance: Series, stance: str) -> float:
        return sum(
            rel / 1
            for i, rel in enumerate(ranking_stance == stance)
        )

    def _rerank_query(self, ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
        ranking["stance_label"].fillna("NO", inplace=True)

        # Boost by inverse discounted gain per stance.
        ranking_stances = set(ranking["stance_label"])
        stance_boost: dict[str, float] = {
            stance: 1 / self._discounted_gain(ranking["stance_label"], stance)
            for stance in self.stances
            if stance in ranking_stances
        }
        stance_boost = defaultdict(lambda: 0.0, stance_boost)
        boost = ranking["stance_label"].map(stance_boost)

        # Normalize scores.
        _normalize_scores(ranking, inplace=True)
        ranking["score"] = (
                (1 - self.alpha) * ranking["score"] +
                self.alpha * boost
        )
        ranking.sort_values("score", ascending=False, inplace=True)
        return ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        groups = ranking.groupby("qid", sort=False, group_keys=False)
        if self.verbose:
            tqdm.pandas(desc="Rerank inverse score frequency", unit="query")
            groups = groups.progress_apply(self._rerank_query)
        else:
            groups = groups.apply(self._rerank_query)
        ranking = groups.reset_index(drop=True)
        ranking = add_ranks(ranking)
        return ranking


@dataclass(frozen=True)
class BoostMinorityStanceFairnessReranker(Transformer):
    boost: float
    stances: Sequence[str] = ("FIRST", "SECOND", "NEUTRAL", "NO")
    verbose: bool = False

    def _rerank_query(self, ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
        ranking["stance_label"].fillna("NO", inplace=True)

        stance_counts = ranking.groupby("stance_label").size().to_dict()
        minority_stance = sorted(
            self.stances,
            key=lambda stance: stance_counts.get(stance, 0)
        )

        # Boost minority stance label.
        stance_boost = {
            stance: self.boost if i == 0 else 1
            for i, stance in enumerate(minority_stance)
        }
        boost = ranking["stance_label"].map(stance_boost)
        ranking["score"] = ranking["score"] * boost
        ranking.sort_values("score", ascending=False, inplace=True)
        return ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        groups = ranking.groupby("qid", sort=False, group_keys=False)
        if self.verbose:
            tqdm.pandas(desc="Rerank boost minority stance", unit="query")
            groups = groups.progress_apply(self._rerank_query)
        else:
            groups = groups.apply(self._rerank_query)
        ranking = groups.reset_index(drop=True)
        ranking = add_ranks(ranking)
        return ranking


class FairnessReranker(Transformer, Enum):
    ORIGINAL = "original"
    INVERSE_STANCE_GAIN = "inverse-stance-gain"
    BOOST_MINORITY_STANCE = "boost-minority-stance"

    @cached_property
    def _transformer(self) -> Transformer:
        if self == FairnessReranker.ORIGINAL:
            return Transformer.identity()
        elif self == FairnessReranker.INVERSE_STANCE_GAIN:
            return InverseStanceGainFairnessReranker(
                stances=("FIRST", "SECOND", "NEUTRAL")
            )
        elif self == FairnessReranker.BOOST_MINORITY_STANCE:
            return BoostMinorityStanceFairnessReranker(boost=2)
        else:
            raise ValueError(f"Unknown fairness re-ranker: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self._transformer.transform(ranking)

    def __repr__(self) -> str:
        return repr(self._transformer)
