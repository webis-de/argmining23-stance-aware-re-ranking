from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from math import isinf, inf, log2
from typing import Sequence, Mapping, AbstractSet

from pandas import DataFrame, Series, read_csv
from pyterrier.model import add_ranks
from pyterrier.transformer import Transformer
from tqdm.auto import tqdm

from stare.utils.stance import stance_value


def _normalize_scores(ranking: DataFrame, inplace: bool = False) -> DataFrame:
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
        ranking["stance_label"].fillna("NO", inplace=True)

        # Boost by inverse discounted gain per stance.
        stance_boost: dict[str, float] = {
            stance: 1 / self._discounted_gain(ranking["stance_label"], stance)
            for stance in self.stances
            if stance in ranking["stance_label"].unique()
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
    allowed_minority_stances: AbstractSet[str] = frozenset(
        {"FIRST", "SECOND", "NEUTRAL", "NO"})
    alpha: float = 0.5
    revision: int = 5

    @cached_property
    def qrels_stance(self) -> DataFrame:
        from stare.config import CONFIG
        qrels = read_csv(
            str(CONFIG.qrels_stance_file_path.absolute()),
            sep="\\s+",
            names=["qid", "0", "docno", "stance_label"],
            dtype=str
        )
        del qrels["0"]
        qrels["stance_value"] = qrels["stance_label"].map(stance_value)
        return qrels

    def _rerank_query(self, ranking: DataFrame) -> DataFrame:
        # Normalize scores to 0-1 without changing ranking order.
        _normalize_scores(ranking, inplace=True)

        # Calculate stance gain.
        discounted_score_gains = (
                ranking["score"] /
                (ranking["rank"] + 1).map(log2)
        )
        discounted_cumulative_score_gain: dict[str, float] = \
            defaultdict(lambda: 0.0)

        for stance_label, gain in zip(
                ranking["stance_label"], discounted_score_gains
                # ranking["stance_label"], ranking["score"]
        ):
            discounted_cumulative_score_gain[stance_label] += gain

        # Determine minority stance.
        ranking["stance_label"].fillna("NO", inplace=True)

        minority_stance = min(
            self.allowed_minority_stances,
            key=lambda stance: discounted_cumulative_score_gain.get(
                stance, inf),
        )

        if minority_stance is None:
            # Could not determine minority.
            return ranking

        # Boost minority stance label.
        stance_boost = {
            stance: 1 if stance == minority_stance else 0
            for stance in ranking["stance_label"].unique()
        }
        boost_scores = ranking["stance_label"].map(stance_boost)

        # Combine scores.
        ranking["score"] = (
                ranking["score"] * (1 - self.alpha) +
                boost_scores * self.alpha
        )

        # Re-order ranking by score.
        ranking.sort_values("score", ascending=False, inplace=True)
        return ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        groups = ranking.groupby("qid", sort=False, group_keys=False)
        ranking = groups.apply(self._rerank_query)
        ranking = ranking.reset_index(drop=True)
        ranking = add_ranks(ranking)
        return ranking


@dataclass(frozen=True)
class InverseStanceFrequencyFairnessReranker(Transformer):
    stance_frequencies: Mapping[str, float] = field(default_factory=lambda: {
        "FIRST": 1,
        "SECOND": 1,
        "NEUTRAL": 1,
        "NO": 1,
    })
    alpha: float = 0.5
    verbose: bool = False

    @cached_property
    def _normalized_stance_frequencies(self) -> dict[str, float]:
        total = sum(
            freq
            for freq in self.stance_frequencies.values()
            if not isinf(freq)
        )
        return {
            stance: freq / total if not isinf(freq) else freq
            for stance, freq in self.stance_frequencies.items()
        }

    def _rerank_query(self, ranking: DataFrame) -> DataFrame:
        # Normalize scores to 0-1 without changing ranking order.
        _normalize_scores(ranking, inplace=True)

        # Compute inverse stance frequency scores.
        ranking["stance_label"].fillna("NO", inplace=True)
        inverse_frequency_scores = ranking["stance_label"] \
            .map(self._normalized_stance_frequencies)

        # Combine scores.
        ranking["score"] = (
                ranking["score"] * (1 - self.alpha) +
                inverse_frequency_scores * self.alpha
        )

        # Re-order ranking by score.
        ranking.sort_values("score", ascending=False, inplace=True)
        return ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        groups = ranking.groupby("qid", sort=False, group_keys=False)
        ranking = groups.apply(self._rerank_query)
        ranking = ranking.reset_index(drop=True)
        ranking = add_ranks(ranking)
        return ranking


class FairnessReranker(Transformer, Enum):
    ORIGINAL = "original"
    INVERSE_STANCE_GAIN = "inverse-stance-gain"
    BOOST_MINORITY_STANCE = "boost-minority-stance"
    INVERSE_STANCE_FREQUENCY = "inverse-stance-frequency"

    @cached_property
    def _transformer(self) -> Transformer:
        if self == FairnessReranker.ORIGINAL:
            return Transformer.identity()
        elif self == FairnessReranker.INVERSE_STANCE_GAIN:
            return InverseStanceGainFairnessReranker(
                stances=("FIRST", "SECOND", "NEUTRAL"),
            )
        elif self == FairnessReranker.BOOST_MINORITY_STANCE:
            return BoostMinorityStanceFairnessReranker(
                allowed_minority_stances={"FIRST", "SECOND", "NEUTRAL"},
            )
        elif self == FairnessReranker.INVERSE_STANCE_FREQUENCY:
            return InverseStanceFrequencyFairnessReranker(
                stance_frequencies={
                    "FIRST": 284,
                    "SECOND": 393,
                    "NEUTRAL": 418,
                    "NO": 1612,
                },
            )
        else:
            raise ValueError(f"Unknown fairness re-ranker: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self._transformer.transform(ranking)

    def __repr__(self) -> str:
        return repr(self._transformer)
