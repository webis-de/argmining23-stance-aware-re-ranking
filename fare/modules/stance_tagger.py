from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from pandas import DataFrame, Series, read_csv
from pyterrier.transformer import Transformer, IdentityTransformer

from fare.api.huggingface import CachedHuggingfaceTextGenerator
from fare.utils.stance import stance_value, stance_label


class T0StanceTagger(Transformer):

    @contextmanager
    def _generator(self) -> CachedHuggingfaceTextGenerator:
        from fare.config import CONFIG
        with CachedHuggingfaceTextGenerator(
                model=CONFIG.huggingface_model_name_t0,
                api_key=CONFIG.huggingface_api_token,
                cache_dir=CONFIG.cache_directory_path,
        ) as generator:
            yield generator

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
            generator: CachedHuggingfaceTextGenerator,
            row: Series,
            object_col: str,
    ) -> float:
        task_pro = self._task_pro(row, object_col)
        answer_pro = generator.generate(task_pro).strip().lower()
        task_con = self._task_con(row, object_col)
        answer_con = generator.generate(task_con).strip().lower()
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
            generator: CachedHuggingfaceTextGenerator,
            row: Series
    ) -> float:
        stance_a = self._stance_single_target(generator, row, "object_first")
        stance_b = self._stance_single_target(generator, row, "object_second")
        return stance_a - stance_b

    def transform(self, ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
        if "stance_value" in ranking.columns:
            ranking.rename({"stance_value": "stance_value_original"})
        if "stance_label" in ranking.columns:
            ranking.rename({"stance_label": "stance_label_original"})

        with self._generator() as generator:
            generator.preload([
                task
                for _, row in ranking.iterrows()
                for task in self._tasks(row)
            ])
            ranking["stance_value"] = [
                self._stance_multi_target(generator, row)
                for _, row in ranking.iterrows()
            ]

        from fare.config import CONFIG

        def threshold_stance_label(value: float) -> str:
            return stance_label(value, CONFIG.stance_filter_threshold)

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
    T0 = "t0"
    GROUND_TRUTH = "ground-truth"

    @cached_property
    def transformer(self) -> Transformer:
        if self == StanceTagger.ORIGINAL:
            return IdentityTransformer()
        elif self == StanceTagger.T0:
            return T0StanceTagger()
        elif self == StanceTagger.GROUND_TRUTH:
            return GroundTruthStanceTagger()
        else:
            raise ValueError(f"Unknown stance tagger: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self.transformer.transform(ranking)
