from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from math import nan, isnan
from pathlib import Path
from statistics import mean

from click import command, argument, Choice
from diskcache import Cache
from nltk import sent_tokenize
from pandas import DataFrame, Series, read_csv
from pyterrier.transformer import Transformer
from sklearn.metrics import f1_score
from torch.cuda import is_available
from tqdm.auto import tqdm
from transformers import (
    Text2TextGenerationPipeline, AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from stare.utils.stance import stance_value, stance_label


@dataclass(frozen=True)
class Text2TextGenerationStanceTagger(Transformer):
    model: str
    verbose: bool = False

    revision: int = 1

    def __post_init__(self):
        from stare.utils.nltk import download_nltk_dependencies
        download_nltk_dependencies("punkt")

    @cached_property
    def _pipeline(self) -> Text2TextGenerationPipeline:
        return Text2TextGenerationPipeline(
            model=AutoModelForSeq2SeqLM.from_pretrained(self.model),
            tokenizer=AutoTokenizer.from_pretrained(self.model),
            device="cuda:0" if is_available() else "cpu"
        )

    @cached_property
    def _cache(self) -> Cache:
        from stare.config import CONFIG
        cache_path = (
                CONFIG.cache_directory_path / "text2text-generation" /
                self.model)
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
class Gpt3TsvStanceTagger(Transformer):
    path: Path = Path("data/stance-gpt-3.tsv")
    qid_column: str = "qid"
    docno_column: str = "ID"
    stance_label_column: str = "gpt_pred_conv"
    fillna: bool = True

    revision: int = 7

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
        ranking = ranking.merge(
            self._tsv_stance,
            how="left",
            on=["qid", "docno"],
            suffixes=("_original", None),
        )
        return ranking


@dataclass(frozen=True)
class RobertaCsvStanceTagger(Transformer):
    path: Path = Path("data/stance-roberta.csv")
    qid_column: str = "Topic"
    docno_column: str = "ID"
    stance_label_column: str = "preds"
    stance_label_map: dict[str, str] = field(
        default_factory=lambda: {
            "0": "NO",
            "1": "NEUTRAL",
            "2": "FIRST",
            "3": "SECOND",
        }
    )
    revision: int = 1

    @cached_property
    def _csv_stance(self) -> DataFrame:
        df = read_csv(
            str(self.path),
            dtype=str
        )
        df = df[[
            self.qid_column,
            self.docno_column,
            self.stance_label_column,
        ]]
        df.rename(columns={
            self.qid_column: "qid",
            self.docno_column: "docno",
            self.stance_label_column: "stance_label",
        }, inplace=True)
        df["stance_label"] = df["stance_label"].map(self.stance_label_map)
        df["stance_value"] = df["stance_label"].map(stance_value)
        return df

    def transform(self, ranking: DataFrame) -> DataFrame:
        ranking = ranking.merge(
            self._csv_stance,
            how="left",
            on=["qid", "docno"],
            suffixes=("_original", None),
        )
        return ranking


@dataclass(frozen=True)
class GroundTruthStanceTagger(Transformer):

    @cached_property
    def qrels_stance(self) -> DataFrame:
        from stare.config import CONFIG
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
        ranking = ranking.merge(
            self.qrels_stance,
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
    LONG_T5_TGLOBAL_BASE = "google/long-t5-tglobal-base"
    GPT3_TSV = "gpt3-tsv"
    ROBERTA_CSV = "roberta-csv"
    GROUND_TRUTH = "ground-truth"

    value: str

    @cached_property
    def _transformer(self) -> Transformer:
        if self == StanceTagger.ORIGINAL:
            return Transformer.identity()
        elif self == StanceTagger.GROUND_TRUTH:
            return GroundTruthStanceTagger()
        elif self == StanceTagger.GPT3_TSV:
            return Gpt3TsvStanceTagger()
        elif self == StanceTagger.ROBERTA_CSV:
            return RobertaCsvStanceTagger()
        elif self in (
                StanceTagger.T0,
                StanceTagger.T0pp,
                StanceTagger.T0_3B,
                StanceTagger.FLAN_T5_BASE,
                StanceTagger.LONG_T5_TGLOBAL_BASE,
        ):
            # noinspection PyTypeChecker
            return Text2TextGenerationStanceTagger(self.value, verbose=True)
        else:
            raise ValueError(f"Unknown stance tagger: {self}")

    def transform(self, ranking: DataFrame) -> DataFrame:
        return self._transformer.transform(ranking)

    def __repr__(self) -> str:
        return repr(self._transformer)


@command()
@argument(
    "tagger",
    type=Choice([str(tagger.value) for tagger in StanceTagger]),
)
def main(tagger: str) -> None:
    from stare.config import CONFIG
    from stare.modules.topics_loader import parse_topics
    from stare.modules.text_loader import TextLoader

    topics = parse_topics()

    qrels = read_csv(
        str(CONFIG.qrels_stance_file_path.absolute()),
        sep="\\s+",
        names=["qid", "0", "docno", "stance_label"],
        dtype=str
    )
    del qrels["0"]
    qrels["stance_value"] = qrels["stance_label"].map(stance_value)

    run_input = qrels.merge(
        topics,
        on="qid",
        how="left",
    )

    pipeline = TextLoader() >> StanceTagger(tagger)
    run = pipeline.transform(run_input)

    df = qrels.merge(
        run,
        on=["qid", "docno"],
        how="left",
        suffixes=("_qrels", "_run"),
    )
    score = f1_score(
        df["stance_label_qrels"],
        df["stance_label_run"],
        average="macro",
    )
    print(f"macro F1: {score}")


if __name__ == '__main__':
    main()
