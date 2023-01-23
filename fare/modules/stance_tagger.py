from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from math import nan, isnan
from pathlib import Path
from statistics import mean
from textwrap import dedent
from time import sleep

import openai
from diskcache import Cache
from nltk import sent_tokenize, word_tokenize
from openai import Completion
from openai.error import RateLimitError
from pandas import DataFrame, Series, read_csv, merge
from pyterrier.transformer import Transformer, IdentityTransformer
from ratelimit import limits, sleep_and_retry
from torch.cuda import is_available
from tqdm.auto import tqdm
from transformers import (
    Pipeline, Text2TextGenerationPipeline, AutoTokenizer,
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    ZeroShotClassificationPipeline
)

from fare import logger
from fare.utils.stance import stance_value, stance_label


@dataclass(frozen=True)
class OpenAiStanceTagger(Transformer):
    model: str = "text-davinci-003"
    verbose: bool = False

    def __post_init__(self):
        from fare.utils.nltk import download_nltk_dependencies
        from fare.config import CONFIG
        download_nltk_dependencies("punkt")
        openai.api_key = CONFIG.open_ai_api_key

    @cached_property
    def _cache(self) -> Cache:
        from fare.config import CONFIG
        cache_path = CONFIG.cache_directory_path / "text-generation" / \
                     "openai" / self.model
        return Cache(str(cache_path))

    @sleep_and_retry
    @limits(calls=1, period=3)
    def _generate_no_cache(self, prompt: str) -> str:
        try:
            response = Completion.create(
                model=self.model,
                prompt=prompt,
                temperature=0.7,
                max_tokens=10,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        except RateLimitError:
            logger.warning("Rate limit exceeded, waiting 10 seconds.")
            sleep(10)
            return self._generate_no_cache(prompt)
        return response.choices[0]["text"]

    def _generate(self, prompt: str) -> str:
        if prompt not in self._cache:
            self._cache[prompt] = self._generate_no_cache(prompt)
        return self._cache[prompt]

    def _stance(self, row: Series) -> float:
        object_first = row["object_first"]
        object_second = row["object_second"]
        question = row["query"]
        document = row["text"]
        prompt = dedent(
            f"""
            Given a comparative question, classify a document's stance as either "pro {object_first}", "pro {object_second}", or "neutral".
            If the document has no personal opinion, recommendation, or pros/cons please return "no stance".
        
            Question: {question}
            Document: {document}
            Stance:
            """
        ).strip()
        label = self._generate(prompt).strip().lower()
        if label.startswith(f"pro {object_first.lower()}"):
            return 1
        elif label.startswith(f"pro {object_second.lower()}"):
            return -1
        elif label.startswith("neutral"):
            return 0
        elif label.startswith("no stance"):
            return nan
        else:
            logger.warning(
                f"Unknown stance label: {label} "
                f"(available: pro {object_first.lower()}, "
                f"pro {object_second.lower()}, neutral, no stance)"
            )
            return nan

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
                desc=f"Tag stance with GPT-3 {self.model}",
                unit="document",
            )
        ranking["stance_value"] = [self._stance(row) for _, row in rows]
        ranking["stance_label"] = ranking["stance_value"].map(stance_label)
        return ranking


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
class TsvStanceTagger(Transformer):
    path: Path
    qid_column: str
    docno_column: str
    stance_label_column: str
    fillna: bool = False

    @cached_property
    def _tsv_stance(self) -> DataFrame:
        df = read_csv(
            str(self.path),
            sep="\t",
            dtype=str
        )
        df = df[[
            self.qid_column,
            self.docno_column,
            self.stance_label_column,
        ]]
        if self.fillna:
            df[self.stance_label_column].fillna("NO", inplace=True)
        else:
            df = df[df[self.stance_label_column].notna()]
        df.rename(columns={
            self.qid_column: "qid",
            self.docno_column: "docno",
            self.stance_label_column: "stance_label",
        }, inplace=True)
        df["stance_value"] = df["stance_label"].map(stance_value)
        return df

    def transform(self, ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
        ranking = merge(
            ranking, self._tsv_stance,
            how="inner",
            on=["qid", "docno"],
            suffixes=("_original", None),
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


@dataclass
class CombinedStanceTagger(Transformer):
    tagger1: Transformer
    tagger2: Transformer
    max_difference: float = 0.5

    @staticmethod
    def _combine(row: Series) -> float:
        stance1: float = row["stance_value_1"]
        stance2: float = row["stance_value_2"]
        if stance1 == stance2:
            return stance1
        elif isnan(stance1):
            return stance2
        elif isnan(stance2):
            return stance1
        elif (stance1 < 0 and stance2 < 0) or (stance1 > 0 and stance2 > 0):
            return max(
                stance1, stance2,
                key=lambda stance: abs(stance)
            )
        else:
            return 0

    def transform(self, ranking: DataFrame) -> DataFrame:
        stance1 = self.tagger1.transform(ranking)
        stance1 = stance1[["qid", "docno", "stance_value"]]
        stance2 = self.tagger2.transform(ranking)
        stance2 = stance2[["qid", "docno", "stance_value"]]
        stance = merge(
            stance1, stance2,
            how="outer",
            on=["qid", "docno"],
            suffixes=("_1", "_2"),
        )
        ranking = ranking.merge(
            stance,
            how="left",
            on=["qid", "docno"],
        )
        ranking["stance_value"] = [
            self._combine(row)
            for _, row in ranking.iterrows()
        ]

        return ranking


class StanceTagger(Transformer, Enum):
    ORIGINAL = "original"
    T0 = "bigscience/T0"
    T0pp = "bigscience/T0pp"
    T0_3B = "bigscience/T0_3B"
    FLAN_T5_BASE = "google/flan-t5-base"
    BART_LARGE_MNLI = "facebook/bart-large-mnli"
    GPT_3_TEXT_DAVINCI_003 = "text-davinci-003"
    FLAN_T5_BASE_GPT_3_TEXT_DAVINCI_003 = "google/flan-t5-base & text-davinci-003"
    GROUND_TRUTH = "ground-truth"

    value: str

    @cached_property
    def _transformer(self) -> Transformer:
        if self == StanceTagger.ORIGINAL:
            return IdentityTransformer()
        elif self == StanceTagger.GROUND_TRUTH:
            return GroundTruthStanceTagger()
        elif self == StanceTagger.FLAN_T5_BASE_GPT_3_TEXT_DAVINCI_003:
            return CombinedStanceTagger(
                TextGenerationStanceTagger(
                    "google/flan-t5-base",
                    verbose=True,
                ),
                OpenAiStanceTagger(
                    "text-davinci-003",
                    verbose=True,
                ),
                max_difference=0.3,
            )
        elif self in (
                StanceTagger.T0,
                StanceTagger.T0pp,
                StanceTagger.T0_3B,
                StanceTagger.FLAN_T5_BASE,
        ):
            # noinspection PyTypeChecker
            return TextGenerationStanceTagger(self.value, verbose=True)
        elif self in (
                StanceTagger.BART_LARGE_MNLI,
        ):
            # noinspection PyTypeChecker
            return ZeroShotClassificationStanceTagger(self.value, verbose=True)
        elif self in (
                StanceTagger.GPT_3_TEXT_DAVINCI_003,
        ):
            # noinspection PyTypeChecker
            return OpenAiStanceTagger(self.value, verbose=True)
        else:
            raise ValueError(f"Unknown stance tagger: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self._transformer.transform(ranking)

    def __repr__(self) -> str:
        return repr(self._transformer)
