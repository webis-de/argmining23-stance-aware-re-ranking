from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from pandas import DataFrame, read_csv
from pyterrier.model import add_ranks
from pyterrier.transformer import Transformer

from stare.utils.stance import stance_value


@dataclass(frozen=True)
class RunLoader(Transformer):
    run_file_path: Path
    _version: int = 1

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
        ranking.sort_values(
            ["qid", "rank"],
            ascending=[True, True],
            inplace=True,
        )
        ranking = add_ranks(ranking)
        return ranking

    @cached_property
    def name(self) -> str:
        return self._ranking["name"].unique()[0]

    @cached_property
    def _transformer(self) -> Transformer:
        return Transformer.from_df(self._ranking)

    def transform(self, topics: DataFrame) -> DataFrame:
        return self._transformer.transform(topics)