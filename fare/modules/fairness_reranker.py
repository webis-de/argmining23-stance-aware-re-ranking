from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from math import nan
from typing import List

from pandas import DataFrame, concat, Series
from pyterrier.model import add_ranks
from pyterrier.transformer import Transformer, IdentityTransformer
from tqdm.auto import tqdm


@dataclass(frozen=True)
class AlternatingStanceReranker(Transformer):
    verbose: bool = False

    @staticmethod
    def _rerank_query(ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
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
        new_ranking = DataFrame(data=new_rows, columns=ranking.columns)

        # Reset rank and score.
        new_ranking["rank"] = list(range(1, len(new_ranking) + 1))
        new_ranking["score"] = list(range(len(new_ranking), 0, -1))

        return new_ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        groups = ranking.groupby("qid", sort=False, group_keys=False)
        if self.verbose:
            tqdm.pandas(desc="Rerank alternating stance", unit="query")
            groups = groups.progress_apply(self._rerank_query)
        else:
            groups = groups.apply(self._rerank_query)
        return groups.reset_index(drop=True)


@dataclass(frozen=True)
class BalancedStanceReranker(Transformer):
    k: int
    verbose: bool = False

    def _rerank_query(self, ranking: DataFrame) -> DataFrame:
        assert 0 <= self.k
        k = min(self.k, len(ranking))

        ranking = ranking.copy().reset_index(drop=True)

        def count_pro_a() -> int:
            head = ranking.iloc[:k]
            return len(head[head["stance_value"] > 0])

        def count_pro_b() -> int:
            head = ranking.iloc[:k]
            return len(head[head["stance_value"] < 0])

        while abs(count_pro_a() - count_pro_b()) > 1:
            # The top-k ranking is currently imbalanced.
            head: DataFrame = ranking.iloc[:k - 1]
            tail: DataFrame = ranking.iloc[k:]

            if count_pro_a() - count_pro_b() > 0:
                # There are currently more documents pro A.
                # Find first pro B document after rank k and
                # move the last pro A document from the top-k ranking
                # behind that document.
                # If no such document is found, we can't balance the ranking.
                candidates_a: DataFrame = head[head["stance_value"] > 0]
                candidates_b: DataFrame = tail[tail["stance_value"] < 0]
                if len(candidates_a) == 0 or len(candidates_b) == 0:
                    return ranking
                else:
                    index_a = candidates_a.index.tolist()[-1]
                    index_b = candidates_b.index.tolist()[0]
                    ranking = concat([
                        ranking.loc[:index_a - 1],
                        ranking.loc[index_a + 1:index_b],
                        ranking.loc[index_a:index_a],
                        ranking.loc[index_b + 1:],
                    ]).reset_index(drop=True)
            else:
                # There are currently more documents pro B.
                # Find first pro A document after rank k and
                # move the last pro B document from the top-k ranking
                # behind that document.
                # If no such document is found,
                # we can't balance the ranking, so return the current ranking.
                candidates_b: DataFrame = head[head["stance_value"] < 0]
                candidates_a: DataFrame = tail[tail["stance_value"] > 0]
                if len(candidates_a) == 0 or len(candidates_b) == 0:
                    return ranking
                else:
                    index_b = candidates_b.index.tolist()[-1]
                    index_a = candidates_a.index.tolist()[0]
                    ranking = concat([
                        ranking.loc[:index_b - 1],
                        ranking.loc[index_b + 1:index_a],
                        ranking.loc[index_b:index_b],
                        ranking.loc[index_a + 1:],
                    ]).reset_index(drop=True)

        # There are equally many documents pro A and pro B.
        # Thus the ranking is already balanced.
        # Return the current ranking.
        return ranking

    def _transform_query(self, ranking: DataFrame) -> DataFrame:
        ranking = self._rerank_query(ranking)

        # Reset rank and score.
        ranking["rank"] = list(range(1, len(ranking) + 1))
        ranking["score"] = list(range(len(ranking), 0, -1))

        return ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        groups = ranking.groupby("qid", sort=False, group_keys=False)
        if self.verbose:
            tqdm.pandas(
                desc=f"Rerank balancing top-{self.k} stance",
                unit="query",
            )
            groups = groups.progress_apply(self._transform_query)
        else:
            groups = groups.apply(self._transform_query)
        return groups.reset_index(drop=True)


@dataclass(frozen=True)
class InverseStanceFrequencyReranker(Transformer):
    verbose: bool = False

    @staticmethod
    def _rerank_query(ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
        ranking_len = len(ranking)
        stance_counts = ranking.groupby("stance_label").size().to_dict()
        stance_frequencies: dict[str, float] = {
            stance: stance_counts[stance] / ranking_len
            if stance in stance_counts else 0
            for stance in ["FIRST", "SECOND", "NEUTRAL", "NO"]
        }

        # Re-score by weighting by inverse stance frequency.
        ranking["score"] = [
            1 / stance_frequencies[row["stance_label"]] * row["score"]
            for _, row in ranking.iterrows()
        ]
        ranking = add_ranks(ranking)
        return ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        groups = ranking.groupby("qid", sort=False, group_keys=False)
        if self.verbose:
            tqdm.pandas(desc="Rerank balance score", unit="query")
            groups = groups.progress_apply(self._rerank_query)
        else:
            groups = groups.apply(self._rerank_query)
        return groups.reset_index(drop=True)


class FairnessReranker(Transformer, Enum):
    ORIGINAL = "original"
    ALTERNATING_STANCE = "alternating-stance"
    BALANCED_TOP_5_STANCE = "balanced-top-5-stance"
    BALANCED_TOP_10_STANCE = "balanced-top-10-stance"
    INVERSE_STANCE_FREQUENCY = "inverse-stance-frequency"

    @cached_property
    def transformer(self) -> Transformer:
        if self == FairnessReranker.ORIGINAL:
            return IdentityTransformer()
        elif self == FairnessReranker.ALTERNATING_STANCE:
            return AlternatingStanceReranker()
        elif self == FairnessReranker.BALANCED_TOP_5_STANCE:
            return BalancedStanceReranker(5, verbose=True)
        elif self == FairnessReranker.BALANCED_TOP_10_STANCE:
            return BalancedStanceReranker(10, verbose=True)
        elif self == FairnessReranker.INVERSE_STANCE_FREQUENCY:
            return InverseStanceFrequencyReranker()
        else:
            raise ValueError(f"Unknown fairness re-ranker: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self.transformer.transform(ranking)
