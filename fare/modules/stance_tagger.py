from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from pandas import DataFrame, Series, read_csv
from pyterrier.transformer import Transformer, IdentityTransformer
from transformers import pipeline, Pipeline

from fare.utils.stance import stance_value, stance_label


@dataclass(frozen=True)
class T0StanceTagger(Transformer):
    model: str

    @cached_property
    def _pipeline(self) -> Pipeline:
        return pipeline(
            task="text-generation",
            model=self.model,
        )

    @staticmethod
    def _task_pro(row: Series, object_col: str) -> str:
        return (
            f"{row['text']}\n\n"
            f"Is this sentence pro {row[object_col]}? yes or no"
        )

    @staticmethod
    def _task_con(row: Series, object_col: str) -> str:
        return (
            f"{row['text']}\n\n"
            f"Is this sentence against {row[object_col]}? yes or no"
        )

    @staticmethod
    def _tasks(row: Series) -> list[str]:
        return [
            T0StanceTagger._task_pro(row, "object_first"),
            T0StanceTagger._task_pro(row, "object_second"),
            T0StanceTagger._task_con(row, "object_first"),
            T0StanceTagger._task_con(row, "object_second"),
        ]

    def _stance_single_target(
            self,
            row: Series,
            object_col: str,
    ) -> float:
        task_pro = self._task_pro(row, object_col)
        answer_pro = self._pipeline(task_pro)[0]["generated_text"] \
            .strip().lower()
        task_con = self._task_con(row, object_col)
        answer_con = self._pipeline(task_con)[0]["generated_text"] \
            .strip().lower()
        is_pro = (
                ("yes" in answer_pro or "pro" in answer_pro) and
                "no" not in answer_pro
        )
        is_con = (
                ("yes" in answer_con or "con" in answer_con) and
                "no" not in answer_con
        )
        if is_pro and not is_con:
            return 1
        elif is_con and not is_pro:
            return -1
        else:
            return 0

    def _stance_multi_target(
            self,
            row: Series
    ) -> float:
        stance_a = self._stance_single_target(row, "object_first")
        stance_b = self._stance_single_target(row, "object_second")
        return stance_a - stance_b

    def transform(self, ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
        if "stance_value" in ranking.columns:
            ranking.rename({"stance_value": "stance_value_original"})
        if "stance_label" in ranking.columns:
            ranking.rename({"stance_label": "stance_label_original"})

        ranking["stance_value"] = [
            self._stance_multi_target(row)
            for _, row in ranking.iterrows()
        ]

        def threshold_stance_label(value: float) -> str:
            return stance_label(value)

        ranking["stance_label"] = ranking["stance_value"].map(
            threshold_stance_label
        )
        return ranking


@dataclass
class GroundTruthStanceTagger(Transformer):

    @cached_property
    def qrels_stance(self) -> DataFrame:
        from fare.config import CONFIG
        qrels = read_csv(
            str(CONFIG.qrels_stance_file_path.absolute()),
            sep="\\s+",
            names=["qid", "0", "docno", "stance_label"]
        )
        del qrels["0"]
        qrels = qrels.astype({"qid": "str"})
        qrels["stance_value"] = qrels["stance_label"].map(stance_value)
        return qrels

    def transform(self, ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
        ranking.merge(
            self.qrels_stance,
            how="left",
            on=["qid", "docno"],
            suffixes=("_original", None)
        )
        return ranking


class StanceTagger(Transformer, Enum):
    ORIGINAL = "original"
    T0 = "bigscience/T0"
    T0pp = "bigscience/T0pp"
    T0_3B = "bigscience/T0_3B"
    GROUND_TRUTH = "ground-truth"

    value: str

    @cached_property
    def transformer(self) -> Transformer:
        if self == StanceTagger.ORIGINAL:
            return IdentityTransformer()
        elif self == StanceTagger.GROUND_TRUTH:
            return GroundTruthStanceTagger()
        elif self in {
            StanceTagger.T0,
            StanceTagger.T0pp,
            StanceTagger.T0_3B,
        }:
            return T0StanceTagger(self.value)
        else:
            raise ValueError(f"Unknown stance tagger: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self.transformer.transform(ranking)
