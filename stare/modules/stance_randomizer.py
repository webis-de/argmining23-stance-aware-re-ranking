from dataclasses import dataclass
from functools import cached_property
from math import log10, ceil
from typing import Optional

from numpy import ndarray
from numpy.random import choice
from pandas import DataFrame, read_csv
from pyterrier.transformer import Transformer
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class StanceF1Randomizer(Transformer):
    max_f1: float
    step: int = 1
    seed: Optional[int] = None

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
        return qrels

    @cached_property
    def qrels_labels(self) -> ndarray:
        return self.qrels_stance["stance_label"].unique()

    @cached_property
    def qrels_frequencies(self) -> ndarray:
        return self.qrels_stance["stance_label"] \
            .value_counts(normalize=True, sort=False) \
            .reindex(self.qrels_labels).values

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if self.max_f1 == 1:
            return topics_or_res

        # Add randomly generated stances.
        topics_or_res["stance_label_random"] = choice(
            self.qrels_labels,
            p=self.qrels_frequencies,
            size=len(topics_or_res),
        )

        # Add qrels stances.
        topics_or_res = topics_or_res.merge(
            self.qrels_stance,
            how="left",
            on=["qid", "docno"],
            suffixes=(None, "_qrels"),
        )

        # Shuffle ranking.
        topics_or_res = topics_or_res.sample(
            frac=1,
            random_state=self.seed,
            ignore_index=True,
        )

        print(f"Randomize stance labels until F1 <= {self.max_f1:.2f}")
        for random_proporiton in range(0, len(topics_or_res), self.step):
            topics_or_res.loc[:random_proporiton, "stance_label"] = \
                topics_or_res.loc[:random_proporiton, "stance_label_random"]
            ranking_eval = topics_or_res \
                .dropna(subset=["stance_label", "stance_label_qrels"])
            f1 = f1_score(
                ranking_eval["stance_label_qrels"].values,
                ranking_eval["stance_label"].values,
                average="macro",
            )
            if random_proporiton % (10 * self.step) == 0:
                print(
                    f"Randomized: "
                    f"{random_proporiton:{ceil(log10(len(topics_or_res)))}d} "
                    f"F1: {f1:.2f}")
            if f1 <= self.max_f1:
                break

        topics_or_res.drop(
            columns=["stance_label_random", "stance_label_qrels"],
            inplace=True,
        )
        return topics_or_res
