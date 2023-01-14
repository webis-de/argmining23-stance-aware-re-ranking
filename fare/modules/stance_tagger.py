from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from math import nan, isnan
from statistics import mean

from diskcache import Cache
from nltk import sent_tokenize, word_tokenize
from pandas import DataFrame, Series, read_csv, merge
from pyterrier.transformer import Transformer, IdentityTransformer
from torch.cuda import is_available
from tqdm.auto import tqdm
from transformers import (
    Pipeline, Text2TextGenerationPipeline, AutoTokenizer,
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    ZeroShotClassificationPipeline
)

from fare.utils.stance import stance_value, stance_label


@dataclass(frozen=True)
class TextGenerationStanceTagger(Transformer):
    model: str
    verbose: bool = False

    def __post_init__(self):
        from fare.utils.nltk import download_nltk_dependencies
        download_nltk_dependencies("punkt")

    @cached_property
    def _pipeline(self) -> Pipeline:
        return Text2TextGenerationPipeline(
            model=AutoModelForSeq2SeqLM.from_pretrained(self.model),
            tokenizer=AutoTokenizer.from_pretrained(self.model),
            device="cuda:0" if is_available() else "cpu"
        )

    @cached_property
    def _cache(self) -> Cache:
        from fare.config import CONFIG
        cache_path = CONFIG.cache_directory_path / "text-generation" / \
                     self.model
        return Cache(str(cache_path))

    def _generate(self, task: str) -> str:
        if task not in self._cache:
            answer = self._pipeline(task)[0]["generated_text"].strip().lower()
            self._cache[task] = answer
        return self._cache[task]

    def _sentence_stance_single_target(
            self,
            sentence: str,
            comparative_object: str,
    ) -> float:
        if comparative_object not in sentence:
            return nan
        task_pro = f"{sentence}\n\n" \
                   f"Is this sentence pro {comparative_object}? yes or no"
        answer_pro = self._generate(task_pro)
        task_con = f"{sentence}\n\n" \
                   f"Is this sentence against {comparative_object}? yes or no"
        answer_con = self._generate(task_con)
        is_pro = (
                ("yes" in answer_pro or "pro" in answer_pro) and
                "no" not in answer_pro
        )
        is_con = (
                ("yes" in answer_con or "con" in answer_con) and
                "no" not in answer_con
        )
        if is_pro and is_con:
            return 0
        elif is_pro and not is_con:
            return 1
        elif is_con and not is_pro:
            return -1
        else:
            return nan

    def _sentence_stance_multi_target(
            self,
            sentence: str,
            object_first: str,
            object_second: str,
    ) -> float:
        stance_a = self._sentence_stance_single_target(sentence, object_first)
        stance_b = self._sentence_stance_single_target(sentence, object_second)
        if isnan(stance_a) and isnan(stance_b):
            return nan
        elif isnan(stance_a):
            return -stance_b
        elif isnan(stance_b):
            return stance_a
        else:
            return stance_a - stance_b

    def _stance_multi_target(self, row: Series) -> float:
        object_first = row["object_first"]
        object_second = row["object_second"]
        sentences = sent_tokenize(row["text"])
        stances = [
            self._sentence_stance_multi_target(
                sentence,
                object_first,
                object_second,
            )
            for sentence in sentences
        ]
        stances = [stance for stance in stances if not isnan(stance)]
        if len(stances) == 0:
            return nan
        return mean(stances)

    def transform(self, ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
        if "stance_value" in ranking.columns:
            ranking.rename({"stance_value": "stance_value_original"})
        if "stance_label" in ranking.columns:
            ranking.rename({"stance_label": "stance_label_original"})

        rows = ranking.iterrows()
        if self.verbose:
            rows = tqdm(
                rows,
                total=len(ranking),
                desc=f"Tag stance with {self.model}",
                unit="document",
            )
        ranking["stance_value"] = [
            self._stance_multi_target(row)
            for _, row in rows
        ]
        ranking["stance_label"] = ranking["stance_value"].map(stance_label)
        return ranking


@dataclass(frozen=True)
class ZeroShotClassificationStanceTagger(Transformer):
    model: str
    verbose: bool = False

    def __post_init__(self):
        from fare.utils.nltk import download_nltk_dependencies
        download_nltk_dependencies("punkt")

    @cached_property
    def _pipeline(self) -> ZeroShotClassificationPipeline:
        return ZeroShotClassificationPipeline(
            model=AutoModelForSequenceClassification.from_pretrained(
                self.model,
            ),
            tokenizer=AutoTokenizer.from_pretrained(self.model),
            device="cuda:0" if is_available() else "cpu",
            multi_label=True,
        )

    @cached_property
    def _cache(self) -> Cache:
        from fare.config import CONFIG
        cache_path = CONFIG.cache_directory_path / "text-classification" / \
                     self.model
        return Cache(str(cache_path))

    def _sentence_stance_multi_target(
            self,
            sentence: str,
            object_first: str,
            object_second: str,
    ) -> float:
        from fare.config import CONFIG

        object_words = {
            *word_tokenize(object_first),
            *word_tokenize(object_second),
        }
        if not any(word in sentence for word in object_words):
            return nan

        cache_key = (sentence, object_first, object_second)
        if cache_key not in self._cache:
            result = self._pipeline(
                sentence,
                candidate_labels=[
                    f"pro {object_first}",
                    f"con {object_first}",
                    f"pro {object_second}",
                    f"con {object_second}",
                ]
            )
            label_scores = {
                label: score
                for label, score in zip(result["labels"], result["scores"])
            }
            self._cache[cache_key] = label_scores
        else:
            label_scores = self._cache[cache_key]

        pro_first = label_scores[f"pro {object_first}"]
        con_first = label_scores[f"con {object_first}"]
        pro_second = label_scores[f"pro {object_second}"]
        con_second = label_scores[f"con {object_second}"]

        threshold = CONFIG.stance_tagger_zero_shot_score_threshold

        stance_first: float
        if pro_first < threshold and con_first < threshold:
            stance_first = nan
        else:
            stance_first = pro_first - con_first

        stance_second: float
        if pro_second < threshold and con_second < threshold:
            stance_second = nan
        else:
            stance_second = pro_second - con_second

        stance: float
        if isnan(stance_first) and isnan(stance_second):
            stance = nan
        elif isnan(stance_first):
            stance = -stance_second
        elif isnan(stance_second):
            stance = stance_first
        else:
            stance = stance_first - stance_second
        return stance

    def _stance_multi_target(self, row: Series) -> float:
        object_first = row["object_first"]
        object_second = row["object_second"]
        sentences = sent_tokenize(row["text"])
        stances = [
            self._sentence_stance_multi_target(
                sentence,
                object_first,
                object_second,
            )
            for sentence in sentences
        ]
        stances = [stance for stance in stances if not isnan(stance)]
        if len(stances) == 0:
            return nan
        stance = mean(stances)
        return stance

    def transform(self, ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
        if "stance_value" in ranking.columns:
            ranking.rename({"stance_value": "stance_value_original"})
        if "stance_label" in ranking.columns:
            ranking.rename({"stance_label": "stance_label_original"})

        rows = ranking.iterrows()
        if self.verbose:
            rows = tqdm(
                rows,
                total=len(ranking),
                desc=f"Tag stance with {self.model}",
                unit="document",
            )
        ranking["stance_value"] = [
            self._stance_multi_target(row)
            for _, row in rows
        ]
        ranking["stance_label"] = ranking["stance_value"].map(stance_label)
        return ranking


@dataclass
class GroundTruthStanceTagger(Transformer):

    @cached_property
    def qrels_stance(self) -> DataFrame:
        from fare.config import CONFIG
        qrels = read_csv(
            str(CONFIG.qrels_stance_file_path.absolute()),
            sep="\\s+",
            names=["qid", "0", "docno", "stance_label"],
            dtype=str
        )
        del qrels["0"]
        qrels["stance_value"] = qrels["stance_label"].map(stance_value)
        return qrels

    def transform(self, ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
        ranking = merge(
            ranking, self.qrels_stance,
            how="left",
            on=["qid", "docno"],
            suffixes=("_original", None),
        )
        return ranking


class StanceTagger(Transformer, Enum):
    ORIGINAL = "original"
    T0 = "bigscience/T0"
    T0pp = "bigscience/T0pp"
    T0_3B = "bigscience/T0_3B"
    FLAN_T5_BASE = "google/flan-t5-base"
    BART_LARGE_MNLI = "facebook/bart-large-mnli"
    GROUND_TRUTH = "ground-truth"

    value: str

    @cached_property
    def transformer(self) -> Transformer:
        if self == StanceTagger.ORIGINAL:
            return IdentityTransformer()
        elif self == StanceTagger.GROUND_TRUTH:
            return GroundTruthStanceTagger()
        elif self in _TEXT_GENERATION_MODELS:
            # noinspection PyTypeChecker
            return TextGenerationStanceTagger(self.value, verbose=True)
        elif self in _ZERO_SHOT_CLASSIFICATION_MODELS:
            # noinspection PyTypeChecker
            return ZeroShotClassificationStanceTagger(self.value, verbose=True)
        else:
            raise ValueError(f"Unknown stance tagger: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self.transformer.transform(ranking)


_TEXT_GENERATION_MODELS = {
    StanceTagger.T0,
    StanceTagger.T0pp,
    StanceTagger.T0_3B,
    StanceTagger.FLAN_T5_BASE
}

_ZERO_SHOT_CLASSIFICATION_MODELS = {
    StanceTagger.BART_LARGE_MNLI
}
