from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from pandas import DataFrame
from pyterrier.transformer import Transformer, IdentityTransformer


class AlternatingStanceReranker(Transformer):

    @staticmethod
    def _alternate_stance(ranking: DataFrame) -> DataFrame:
        old_ranking = ranking.copy()
        new_ranking = []

        last_stance: float = 0
        while len(old_ranking) > 0:
            index: int

            if last_stance > 0:
                # Last document was pro A.
                # Find first pro B or neutral document next.
                # If no such document is found, choose
                # first document regardless of stance.
                index = next(
                    (
                        i
                        for i, document in enumerate(old_ranking)
                        if _stance(document) <= 0
                    ),
                    0
                )
            elif last_stance < 0:
                # Last document was pro B.
                # Find first pro A or neutral document next.
                # If no such document is found, choose
                # first document regardless of stance.
                index = next(
                    (
                        i
                        for i, document in enumerate(old_ranking)
                        if _stance(document) >= 0
                    ),
                    0
                )
            else:
                # Last document was neutral.
                # Find any document next, regardless of stance.
                index = 0

            document = old_ranking.pop(index)
            new_ranking.append(document)
            last_stance = _stance(document)

        return new_ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        ranking = self._alternate_stance(ranking)
        ranking = _reset_score(ranking)
        return ranking


@dataclass
class BalancedStanceReranker(Transformer):
    k: int

    def _balanced_top_k_stance(self, ranking: DataFrame) -> DataFrame:
        assert 0 <= self.k
        k = min(self.k, len(ranking))

        ranking = ranking.copy()

        def count_pro_a() -> int:
            return sum(
                1 for document in ranking[:k]
                if _stance(document) > 0
            )

        def count_pro_b() -> int:
            return sum(
                1 for document in ranking[:k]
                if _stance(document) < 0
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
                    for i in range(k + 1)
                    if _stance(ranking[i]) > 0
                ), None)
                index_b = next((
                    i
                    for i in range(k + 1, len(ranking))
                    if _stance(ranking[i]) < 0
                ), None)
                if index_a is None or index_b is None:
                    return ranking
                else:
                    document_a = ranking.pop(index_a)
                    # Pro B document has moved one rank up now.
                    ranking.insert(index_b, document_a)
            else:
                # There are currently more documents pro B.
                # Find first pro A document after rank k and
                # move the last pro B document from the top-k ranking
                # behind that document.
                # If no such document is found,
                # we can't balance the ranking, so return the current ranking.
                index_b = next((
                    i
                    for i in range(k + 1)
                    if _stance(ranking[i]) < 0
                ), None)
                index_a = next((
                    i
                    for i in range(k + 1, len(ranking))
                    if _stance(ranking[i]) > 0
                ), None)
                if index_b is None or index_a is None:
                    return ranking
                else:
                    document_b = ranking.pop(index_b)
                    # Pro A document has moved one rank up now.
                    ranking.insert(index_a, document_b)

        # There are equally many documents pro A and pro B.
        # Thus the ranking is already balanced.
        # Return the current ranking.
        return ranking

    def transform(self, ranking: DataFrame) -> DataFrame:
        ranking = self._balanced_top_k_stance(ranking)
        ranking = _reset_score(ranking)
        return ranking



class FairnessReranker(Transformer, Enum):
    ORIGINAL = "original"
    ALTERNATING_STANCE = "alternating-stance"
    BALANCED_STANCE = "balanced-stance"

    @cached_property
    def transformer(self) -> Transformer:
        if self == FairnessReranker.ORIGINAL:
            return IdentityTransformer()
        elif self == FairnessReranker.ALTERNATING_STANCE:
            return AlternatingStanceReranker()
        elif self == FairnessReranker.BALANCED_STANCE:
            return BalancedStanceReranker()
        else:
            raise ValueError(f"Unknown fairness re-ranker: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self.transformer.transform(ranking)
