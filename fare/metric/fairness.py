"""
Fairness measures from "Evaluating Fairness in Argument Retrieval"
Paper: https://doi.org/10.1145/3459637.3482099
Code: https://github.com/sachinpc1993/fair-arguments
"""
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from math import log
from typing import Iterable, Union, Iterator, Hashable, Optional, Final

from ir_measures import DefaultPipeline
from ir_measures.measures import Measure, ParamInfo
from ir_measures.providers import Provider, Evaluator, register, registry
from ir_measures.util import (
    flatten_measures, QrelsConverter, RunConverter, Qrel, ScoredDoc, Metric
)
from pandas import DataFrame

Qrels = Union[Iterable[Qrel], dict[str, dict[str, int]], DataFrame]
Run = Union[Iterable[ScoredDoc], dict[str, dict[str, int]], DataFrame]


class ProtectedGroupSelection(Enum):
    MINORITY = "minority"
    MAJORITY = "majority"


class FairnessMeasure(Measure, ABC):
    SUPPORTED_PARAMS = {
        "cutoff": ParamInfo(
            dtype=int,
            required=False,
            desc="ranking cutoff threshold"
        ),
        "group_col": ParamInfo(
            dtype=Hashable,
            default="group",
            desc="group column in run"
        ),
        "groups": ParamInfo(
            dtype=set[Hashable],
            required=False,
            desc="group names"
        ),
        "protected_group": ParamInfo(
            dtype=Hashable,
            default=ProtectedGroupSelection.MINORITY,
            desc="protected group name or selection strategy"
        ),
    }

    @cached_property
    def _cutoff_param(self) -> Optional[int]:
        if "cutoff" not in self.params:
            return None
        return self.params["cutoff"]

    @cached_property
    def _group_col_param(self) -> Hashable:
        if "group_col" not in self.params:
            return "group"
        return self.params["group_col"]

    @cached_property
    def _groups_param(self) -> Optional[set[Hashable]]:
        if "groups" not in self.params:
            return None
        return self.params["groups"]

    @cached_property
    def _protected_group_param(self) -> Union[
        ProtectedGroupSelection, Hashable
    ]:
        if "protected_group" not in self.params:
            return ProtectedGroupSelection.MINORITY
        protected_group = self.params["protected_group"]
        if protected_group in {s.value for s in ProtectedGroupSelection}:
            return ProtectedGroupSelection(protected_group)
        return protected_group

    @abstractmethod
    def fairness(
            self,
            ranking: DataFrame,
            group_col: Hashable,
            group_counts: dict[Hashable, int],
            protected_group: Hashable,
    ) -> float:
        pass

    def _compute_query(
            self,
            qrels: DataFrame,
            ranking: DataFrame,
            groups: set[Hashable],
    ) -> float:
        group_counts = self._group_counts(qrels, groups)
        protected_group = self._protected_group(group_counts)
        if protected_group not in groups:
            raise ValueError(
                f"Protected group {protected_group} "
                f"not found in groups {groups}."
            )
        return self.fairness(
            ranking,
            self._group_col_param,
            group_counts,
            protected_group,
        )

    def _groups(self, qrels: DataFrame) -> set[Hashable]:
        if self._groups_param is not None:
            return self._groups_param
        return set(qrels[self._group_col_param].unique().tolist())

    def _group_counts(
            self,
            qrels: DataFrame,
            groups: set[Hashable],
    ) -> dict[Hashable, int]:
        counts = qrels.groupby(self._group_col_param).size().to_dict()
        return {
            group: counts.get(group, 0)
            for group in groups
        }

    def _protected_group(self, group_counts: dict[Hashable, int]) -> Hashable:
        protected_group: Hashable = self._protected_group_param
        if isinstance(protected_group, ProtectedGroupSelection):
            groups: list[tuple[Hashable, int]] = [
                item for item in group_counts.items()
            ]
            if protected_group == ProtectedGroupSelection.MINORITY:
                return sorted(groups, key=lambda item: item[1])[0][0]
            elif protected_group == ProtectedGroupSelection.MAJORITY:
                return sorted(groups, key=lambda item: item[1])[-1][0]
            else:
                raise ValueError()
        else:
            return protected_group

    def compute(self, qrels: DataFrame, run: DataFrame) -> Iterator[Metric]:
        groups = self._groups(qrels)

        if self._cutoff_param is not None:
            # Assumes that results are already sorted.
            # (This is done in FairnessEvaluator.)
            run = run.groupby("query_id").head(self._cutoff_param).reset_index(
                drop=True)

        for qid, ranking in run.groupby("query_id"):
            qid: str
            ranking_qrels = qrels[qrels["query_id"] == qid]
            yield Metric(
                qid,
                self,
                self._compute_query(ranking_qrels, ranking, groups)
            )

    def __str__(self):
        name = self.NAME
        cutoff = (
            f"@{self._cutoff_param}"
            if self._cutoff_param is not None
            else ""
        )
        protected_group = (
            self._protected_group_param.value
            if isinstance(self._protected_group_param, ProtectedGroupSelection)
            else repr(self._protected_group_param)
        )
        params = [
            param
            for param in (self._groups_param, protected_group)
            if param is not None
        ]
        return f"{name}{cutoff}({','.join(params)})"


class _NormalizedDiscountedDifference(FairnessMeasure):
    NAME = "rND"
    __name__ = "rND"

    def fairness(
            self,
            ranking: DataFrame,
            group_col: Hashable,
            group_counts: dict[Hashable, int],
            protected_group: Hashable,
    ) -> float:
        ranking_df = ranking

        N = sum(group_counts.values())

        rnd_list = []

        # For each ranking position
        for i in range(1, 6):
            temp_ranking = ranking_df[ranking_df["rank"].isin(range(1, i + 1))]

            stance_freq = temp_ranking[group_col].value_counts()
            stance_freq_df = stance_freq.to_frame().reset_index().rename(
                columns={
                    group_col: "count",
                    "index": "group"
                })

            if len(stance_freq_df[
                       stance_freq_df["group"] == protected_group
                   ]["count"]) == 0:
                S_Plus_in_i = 0
            else:
                S_Plus_in_i = list(stance_freq_df[
                                       stance_freq_df[
                                           "group"] == protected_group
                                       ]["count"])[0]

            S_plus = group_counts[protected_group]

            intermediate_rnd = (
                    (1 / log(i + 1, 2)) *
                    abs(abs(S_Plus_in_i / (i + 1)) - abs(S_plus / N))
            )

            rnd_list.append(intermediate_rnd)

        final_rnd = sum(rnd_list)

        return final_rnd


NormalizedDiscountedDifference = _NormalizedDiscountedDifference()
rND = NormalizedDiscountedDifference


@dataclass
class FairnessEvaluator(Evaluator):
    measures: Final[Iterable[FairnessMeasure]] = field()
    qrels: DataFrame

    def __post_init__(self):
        super().__init__(self.measures, set(self.qrels["query_id"].unique()))

    def _iter_calc(self, run: Run) -> Iterator[Metric]:
        run: DataFrame = RunConverter(run).as_pd_dataframe()
        run.sort_values(
            by=["query_id", "score"],
            ascending=[True, False],
            inplace=True,
        )
        for measure in self.measures:
            yield from measure.compute(self.qrels, run)


@dataclass(frozen=True)
class FairnessProvider(Provider):
    NAME = "fairness"
    SUPPORTED_MEASURES = [
        rND
    ]
    _is_available = True

    def _evaluator(
            self,
            measures: Iterable[FairnessMeasure],
            qrels: Qrels
    ) -> FairnessEvaluator:
        measures = flatten_measures(measures)
        qrels: DataFrame = QrelsConverter(qrels).as_pd_dataframe()
        qrels.sort_values(
            by=["query_id", "doc_id"],
            inplace=True,
        )
        return FairnessEvaluator(measures, qrels)


register(FairnessProvider())
DefaultPipeline.providers.append(registry["fairness"])
