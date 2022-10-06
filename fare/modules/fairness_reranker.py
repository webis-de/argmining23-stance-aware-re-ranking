from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from math import nan

from pandas import DataFrame, concat
from pyterrier.transformer import Transformer, IdentityTransformer
from tqdm.auto import tqdm


@dataclass(frozen=True)
class AlternatingStanceReranker(Transformer):
    verbose: bool = False

    @staticmethod
    def _rerank_query(ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
        new_rows = []

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
                # No cadidate for the stance was found,
                # choose any document next,
                # regardless of stance.
                last_stance = nan
                continue

            document = ranking.loc[candidates.index].iloc[0]
            last_stance = document["stance_value"]
            new_rows.append(document)
            ranking.drop(index=candidates.index, inplace=True)
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

        ranking = ranking.copy()

        def count_pro_a() -> int:
            return sum(
                1 for _, row in ranking.iloc[:k].iterrows()
                if row["stance_value"] > 0
            )

        def count_pro_b() -> int:
            return sum(
                1 for _, row in ranking.iloc[:k].iterrows()
                if row["stance_value"] < 0
            )

        while abs(count_pro_a() - count_pro_b()) > 1:
            # The top-k ranking is currently imbalanced.

            if count_pro_a() - count_pro_b() > 0:
                # There are currently more documents pro A.
                # Find first pro B document after rank k and
                # move the last pro A document from the top-k ranking
                # behind that document.
                # If no such document is found, we can't balance the ranking.
                index_a = next((
                    i
                    for i, row in ranking.iloc[:k].iterrows()
                    if row["stance_value"] > 0
                ), None)
                index_b = next((
                    i
                    for i, row in ranking.iloc[k + 1:].iterrows()
                    if row["stance_value"] < 0
                ), None)
                if index_a is None or index_b is None:
                    return ranking
                else:
                    ranking = concat([
                        ranking.iloc[:index_a - 1],
                        ranking.iloc[index_a + 1:index_b],
                        ranking.iloc[index_a:index_a],
                        ranking.iloc[index_b + 1:],
                    ])
            else:
                # There are currently more documents pro B.
                # Find first pro A document after rank k and
                # move the last pro B document from the top-k ranking
                # behind that document.
                # If no such document is found,
                # we can't balance the ranking, so return the current ranking.
                index_b = next((
                    i
                    for i, row in ranking.iloc[:k].iterrows()
                    if row["stance_value"] < 0
                ), None)
                index_a = next((
                    i
                    for i, row in ranking.iloc[k + 1:].iterrows()
                    if row["stance_value"] > 0
                ), None)
                if index_b is None or index_a is None:
                    return ranking
                else:
                    ranking = concat([
                        ranking.iloc[:index_b - 1],
                        ranking.iloc[index_b + 1:index_a],
                        ranking.iloc[index_b:index_b],
                        ranking.iloc[index_a + 1:],
                    ])

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


class FairnessReranker(Transformer, Enum):
    ORIGINAL = "original"
    ALTERNATING_STANCE = "alternating-stance"
    BALANCED_TOP_5_STANCE = "balanced-top-5-stance"
    BALANCED_TOP_10_STANCE = "balanced-top-10-stance"

    @cached_property
    def transformer(self) -> Transformer:
        if self == FairnessReranker.ORIGINAL:
            return IdentityTransformer()
        elif self == FairnessReranker.ALTERNATING_STANCE:
            return AlternatingStanceReranker(verbose=True)
        elif self == FairnessReranker.BALANCED_TOP_5_STANCE:
            return BalancedStanceReranker(5, verbose=True)
        elif self == FairnessReranker.BALANCED_TOP_10_STANCE:
            return BalancedStanceReranker(10, verbose=True)
        else:
            raise ValueError(f"Unknown fairness re-ranker: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self.transformer.transform(ranking)
