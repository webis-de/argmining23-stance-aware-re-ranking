from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from pandas import DataFrame, read_csv
from pyterrier.transformer import Transformer

from fare.utils.stance import stance_value


@dataclass(frozen=True)
class RunLoader(Transformer):
    run_file_path: Path

    @cached_property
    def _ranking(self) -> DataFrame:
        ranking = read_csv(
            str(self.run_file_path.absolute()),
            sep="\\s+",
            names=["qid", "stance_label", "docno", "rank", "score", "name"],
        )
        ranking = ranking.astype({"qid": "str"})
        ranking["stance_label"] = ranking["stance_label"].replace("Q0", "NO")
        ranking["stance_value"] = ranking["stance_label"].map(stance_value)
        return ranking

    @cached_property
    def name(self) -> str:
        return self._ranking["name"].unique()[0]

    @cached_property
    def _transformer(self) -> Transformer:
        return Transformer.from_df(self._ranking)

    def transform(self, topics: DataFrame) -> DataFrame:
        return self._transformer.transform(topics)
