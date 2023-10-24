from dataclasses import dataclass

from pandas import DataFrame
from pyterrier.transformer import Transformer


@dataclass(frozen=True)
class StanceFilter(Transformer):
    threshold: float = 0.5

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        topics_or_res.loc[
            topics_or_res["stance_value"].abs() < self.threshold,
            "stance_value"
        ] = 0
        topics_or_res.loc[
            topics_or_res["stance_value"] == 0,
            "stance_label"
        ] = "NEUTRAL"
        return topics_or_res
