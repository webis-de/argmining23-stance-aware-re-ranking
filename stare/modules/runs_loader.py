from dataclasses import dataclass
from functools import cached_property, cache
from pathlib import Path

from pandas import DataFrame, read_csv
from pyterrier.model import add_ranks
from pyterrier.transformer import Transformer

from stare.utils.stance import stance_value


@cache
def _read_run(run_file_path: Path) -> DataFrame:
    ranking = read_csv(
        str(run_file_path.absolute()),
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


@dataclass(frozen=True)
class RunLoader(Transformer):
    run_file_path: Path
    _version: int = 1

    @cached_property
    def _ranking(self) -> DataFrame:
        return _read_run(self.run_file_path)

    @cached_property
    def name(self) -> str:
        return self._ranking["name"].unique()[0]

    @cached_property
    def _transformer(self) -> Transformer:
        return Transformer.from_df(self._ranking)

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return self._transformer.transform(topics_or_res)
