from dataclasses import dataclass
from functools import cached_property
from math import log10, ceil

from numpy.random import choice
from pandas import DataFrame, read_csv
from pyterrier.transformer import Transformer
from sklearn.metrics import f1_score

from fare.utils.stance import stance_label, stance_value


class StanceFractionRandomizeMethod:
    DRAW = "draw"
    FLIP = "flip"


@dataclass(frozen=True)
class StanceFractionRandomizer(Transformer):
    method: StanceFractionRandomizeMethod
    fraction: float

    revision: int = 2

    def transform(self, ranking: DataFrame) -> DataFrame:
        if self.fraction == 0:
            return ranking

        ranking = ranking.copy()
        ranking_stance_notna = ranking[ranking["stance_label"].notna()]
        to_be_randomized = ranking_stance_notna.sample(
            frac=self.fraction,
            random_state=0,
        ).index

        if self.method == StanceFractionRandomizeMethod.FLIP:
            ranking.loc[to_be_randomized, "stance_value"] *= -1
            ranking.loc[to_be_randomized, "stance_label"] = \
                ranking.loc[to_be_randomized, "stance_value"].map(stance_label)
        elif self.method == StanceFractionRandomizeMethod.DRAW:
            labels = ranking.loc[to_be_randomized, "stance_label"].unique()
            frequencies = ranking.loc[to_be_randomized, "stance_label"] \
                .value_counts(normalize=True, sort=False) \
                .reindex(labels).values
            ranking.loc[to_be_randomized, "stance_label"] = choice(
                labels,
                p=frequencies,
                size=len(to_be_randomized),
            )
            ranking.loc[to_be_randomized, "stance_value"] = \
                ranking.loc[to_be_randomized, "stance_label"].map(stance_value)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return ranking


@dataclass(frozen=True)
class StanceF1Randomizer(Transformer):
    max_f1: float
    step: int = 100

    revision: int = 2

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
        return qrels

    def transform(self, ranking: DataFrame) -> DataFrame:
        if self.max_f1 == 1:
            return ranking

        ranking = ranking.copy()

        # Add randomly generated stances.
        labels = self.qrels_stance["stance_label"].unique()
        frequencies = self.qrels_stance["stance_label"] \
            .value_counts(normalize=True, sort=False) \
            .reindex(labels).values
        ranking["stance_label_random"] = choice(
            labels,
            p=frequencies,
            size=len(ranking),
        )

        # Add qrels stances.
        ranking = ranking.merge(
            self.qrels_stance,
            how="left",
            on=["qid", "docno"],
            suffixes=(None, "_qrels"),
        )

        # Shuffle ranking.
        ranking = ranking.sample(frac=1, random_state=0, ignore_index=True)

        print(f"Randomize stance labels until F1 <= {self.max_f1:.2f}")
        for cutoff in range(0, len(ranking), self.step):
            ranking.loc[:cutoff, "stance_label"] = \
                ranking.loc[:cutoff, "stance_label_random"]
            ranking_eval = ranking \
                .dropna(subset=["stance_label", "stance_label_qrels"])
            f1 = f1_score(
                ranking_eval["stance_label_qrels"].values,
                ranking_eval["stance_label"].values,
                average="macro",
            )
            if cutoff % (10 * self.step) == 0:
                print(f"Randomized: {cutoff:{ceil(log10(len(ranking)))}d} "
                      f"F1: {f1:.2f}")
            if f1 <= self.max_f1:
                break

        ranking.drop(
            columns=["stance_label_random", "stance_label_qrels"],
            inplace=True,
        )
        return ranking
