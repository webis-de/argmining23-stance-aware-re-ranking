from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from pandas import DataFrame, read_csv
from pyterrier.transformer import Transformer

from stare.utils.pyterrier import reset_order


@dataclass(frozen=True)
class OptimalRelevanceReranker(Transformer):
    revision: int = 7

    @cached_property
    def qrels(self) -> DataFrame:
        from stare.config import CONFIG
        return read_csv(
            str(CONFIG.qrels_relevance_file_path.absolute()),
            sep="\\s+",
            names=["qid", "0", "docno", "label"],
            dtype=str
        ).drop(columns=["0"])

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        topics_or_res = topics_or_res.merge(
            self.qrels,
            how="left",
            on=["qid", "docno"],
        ).sort_values(
            by=["qid", "label"],
            ascending=[True, False],
        ).drop(
            columns=["label"],
        )
        topics_or_res = reset_order(topics_or_res)
        return topics_or_res


@dataclass(frozen=True)
class OptimalQualityReranker(Transformer):
    revision: int = 7

    @cached_property
    def qrels(self) -> DataFrame:
        from stare.config import CONFIG
        return read_csv(
            str(CONFIG.qrels_quality_file_path.absolute()),
            sep="\\s+",
            names=["qid", "0", "docno", "label"],
            dtype=str
        ).drop(columns=["0"])

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        topics_or_res = topics_or_res.merge(
            self.qrels,
            how="left",
            on=["qid", "docno"],
        ).sort_values(
            by=["qid", "label"],
            ascending=[True, False],
        ).drop(
            columns=["label"],
        )
        topics_or_res = reset_order(topics_or_res)
        return topics_or_res


class OptimalReranker(Transformer, Enum):
    RELEVANCE = "optimal-relevance"
    QUALITY = "optimal-quality"

    value: str

    @cached_property
    def _transformer(self) -> Transformer:
        if self == OptimalReranker.RELEVANCE:
            return OptimalRelevanceReranker()
        elif self == OptimalReranker.QUALITY:
            return OptimalQualityReranker()
        else:
            raise ValueError(f"Unknown optimal re-ranker: {self}")

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return self._transformer.transform(topics_or_res)

    def __repr__(self) -> str:
        return repr(self._transformer)
