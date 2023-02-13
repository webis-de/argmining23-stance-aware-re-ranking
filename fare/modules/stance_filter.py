from dataclasses import dataclass

from pandas import DataFrame
from pyterrier.transformer import Transformer


@dataclass(frozen=True)
class StanceFilter(Transformer):
    threshold: float = 0.5

    def transform(self, ranking: DataFrame) -> DataFrame:
        ranking.loc[
            ranking["stance_value"].abs() < self.threshold,
            "stance_value"
        ] = 0
        ranking.loc[
            ranking["stance_value"] == 0,
            "stance_label"
        ] = "NEUTRAL"
        return ranking
