"""
Fairness measures from:
- Measuring Fairness in Ranked Outputs: https://doi.org/10.1145/3085504.3085526 https://github.com/DataResponsibly/FairRank
- Evaluating Fairness in Argument Retrieval: https://doi.org/10.1145/3459637.3482099 https://github.com/sachinpc1993/fair-arguments
"""
from abc import abstractmethod, ABC
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from itertools import permutations
from math import log, nextafter
from random import choice
from typing import (
    Iterable, Union, Iterator, Hashable, Optional, Final, Sequence
)

from ir_measures import DefaultPipeline
from ir_measures.measures import (
    Measure, ParamInfo, register as register_measure
)
from ir_measures.providers import (
    Provider, Evaluator, register as register_provider
)
from ir_measures.util import (
    flatten_measures, QrelsConverter, RunConverter, Qrel, ScoredDoc, Metric
)
from pandas import DataFrame
from scipy.special import rel_entr

Qrels = Union[Iterable[Qrel], dict[str, dict[str, int]], DataFrame]
Run = Union[Iterable[ScoredDoc], dict[str, dict[str, int]], DataFrame]


class ProtectedGroupStrategy(Enum):
    MINORITY = "minority"
    MAJORITY = "majority"


class TieBreakingStrategy(Enum):
    RANDOM = "random"
    GROUP_ASCENDING = "group-ascending"
    GROUP_DESCENDING = "group-descending"


class FairnessMeasure(Measure, ABC):
    SUPPORTED_PARAMS = {
        "cutoff": ParamInfo(
            dtype=int,
            required=False,
            desc="ranking cutoff threshold"
        ),
        "group_col": ParamInfo(
            dtype=Hashable,
            required=False,
            default="group",
            desc="group column in run"
        ),
        "groups": ParamInfo(
            dtype=str,
            required=False,
            desc="comma-separated list of group names"
        ),
        "protected_group": ParamInfo(
            dtype=Hashable,
            required=False,
            default=ProtectedGroupStrategy.MINORITY.value,
            desc="protected group name or selection strategy"
        ),
        "tie_breaking": ParamInfo(
            dtype=Hashable,
            required=False,
            desc="tie breaking strategy when selecting the protected group "
                 "using the minority or majority strategies or a "
                 "comma-separated preference list"
                 "(if not specified, ties will raise an exception)"
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
    def _groups_param(self) -> Optional[set[str]]:
        if "groups" not in self.params:
            return None
        groups = self.params["groups"]
        return {
            group.strip()
            for group in str(groups).split(",")
        }

    @cached_property
    def _protected_group_param(self) -> Union[
        ProtectedGroupStrategy, Hashable
    ]:
        if "protected_group" not in self.params:
            return ProtectedGroupStrategy.MINORITY
        protected_group = self.params["protected_group"]
        if protected_group in {s.value for s in ProtectedGroupStrategy}:
            return ProtectedGroupStrategy(protected_group)
        if not isinstance(protected_group, Hashable):
            raise ValueError(
                f"Illegal protected_group param: {protected_group}"
            )
        return protected_group

    @cached_property
    def _tie_breaking_param(self) -> Union[
        TieBreakingStrategy, Sequence, None
    ]:
        if "tie_breaking" not in self.params:
            return None
        tie_breaking = self.params["tie_breaking"]
        if tie_breaking is None:
            return None
        if tie_breaking in {s.value for s in TieBreakingStrategy}:
            return TieBreakingStrategy(tie_breaking)
        if not isinstance(tie_breaking, Hashable):
            raise ValueError(
                f"Illegal tie_breaking param: {tie_breaking}"
            )
        return [
            group.strip()
            for group in str(tie_breaking).split(",")
        ]

    @abstractmethod
    def unfairness(
            self,
            ranking: tuple[Hashable],
            groups: set[Hashable],
            group_counts: dict[Hashable, int],
            protected_group: Hashable,
    ) -> float:
        pass

    @staticmethod
    def group_counts(
            qrels_or_ranking: tuple[Hashable],
            groups: set[Hashable],
    ) -> dict[Hashable, int]:
        counts = Counter(qrels_or_ranking)
        return {group: counts.get(group, 0) for group in groups}

    def _protected_group(self, group_counts: dict[Hashable, int]) -> Hashable:
        protected_group: Hashable = self._protected_group_param
        if isinstance(protected_group, ProtectedGroupStrategy):
            strategy: ProtectedGroupStrategy = protected_group
            groups: list[tuple[Hashable, int]] = [
                item for item in group_counts.items()
            ]
            if strategy == ProtectedGroupStrategy.MINORITY:
                groups = sorted(groups, key=lambda item: item[1])
            elif strategy == ProtectedGroupStrategy.MAJORITY:
                groups = sorted(groups, key=lambda item: item[1], reverse=True)
            else:
                raise ValueError(
                    f"Unknown protected group strategy: {strategy}"
                )
            if len(groups) > 1 and groups[0][1] == groups[1][1]:
                # Tie in group selection.
                count = groups[0][1]
                tie_groups = [
                    group[0]
                    for group in groups
                    if group[1] == count
                ]
                tie_breaking = self._tie_breaking_param
                if tie_breaking is None:
                    raise ValueError(
                        f"Could not select protected group "
                        f"by {strategy.value} because of a tie. "
                        f"Groups {tie_groups} all occur {count} time(s)."
                    )
                elif isinstance(tie_breaking, TieBreakingStrategy):
                    if not all(hasattr(g, "__lt__") for g in tie_groups):
                        raise ValueError(
                            f"Tie breaking {tie_breaking.value} requires "
                            f"sorting but groups are not "
                            f"sortable: {tie_groups}"
                        )
                    if tie_breaking == TieBreakingStrategy.RANDOM:
                        return choice(tie_groups)
                    elif tie_breaking == TieBreakingStrategy.GROUP_ASCENDING:
                        # noinspection PyTypeChecker
                        return sorted(tie_groups)[0]
                    elif tie_breaking == TieBreakingStrategy.GROUP_DESCENDING:
                        # noinspection PyTypeChecker
                        return sorted(tie_groups, reverse=True)[0]
                else:
                    tie_breaking_groups = [
                        group
                        for group in tie_breaking
                        if group in tie_groups
                    ]
                    if len(tie_breaking_groups) == 0:
                        raise ValueError(
                            f"Tie breaking preference {tie_breaking} not "
                            f"applicable to resolve tie: {tie_groups}"
                        )
                    return tie_breaking_groups[0]
            return groups[0][0]
        else:
            return protected_group

    @staticmethod
    def _permuted_rankings(ranking: tuple[Hashable]) -> set[tuple[Hashable]]:
        # noinspection PyTypeChecker
        return set(permutations(ranking))

    def max_unfairness(
            self,
            ranking: tuple[Hashable],
            groups: set[Hashable],
            group_counts: dict[Hashable, int],
            protected_group: Hashable,
    ) -> float:
        return max(
            self.unfairness(
                permuted_ranking,
                groups,
                group_counts,
                protected_group,
            )
            for permuted_ranking in self._permuted_rankings(ranking)
        )

    def _compute_query(
            self,
            qrels: tuple[Hashable],
            ranking: tuple[Hashable],
            groups: set[Hashable],
    ) -> float:
        group_counts = self.group_counts(qrels, groups)
        protected_group = self._protected_group(group_counts)
        if protected_group not in group_counts.keys():
            raise ValueError(
                f"Protected group {protected_group} "
                f"not found in groups {set(group_counts.keys())}."
            )

        max_unfairness = self.max_unfairness(
            ranking,
            groups,
            group_counts,
            protected_group
        )
        if max_unfairness == 0:
            return 0
        unfairness = self.unfairness(
            ranking,
            groups,
            group_counts,
            protected_group
        )
        normalized_unfairness = unfairness / max_unfairness
        return normalized_unfairness

    def _groups(self, qrels_or_ranking: tuple[Hashable]) -> set[Hashable]:
        if self._groups_param is not None:
            return self._groups_param
        return set(qrels_or_ranking)

    def compute(self, qrels: DataFrame, run: DataFrame) -> Iterator[Metric]:
        group_col = self._group_col_param
        qrels = qrels[["query_id", group_col]]
        run = run[["query_id", group_col]]

        groups = self._groups(qrels[group_col])
        if len(groups) == 0:
            raise ValueError("No groups given.")

        if self._cutoff_param is not None:
            run = run.groupby("query_id").head(self._cutoff_param)

        for qid, ranking in run.groupby("query_id"):
            yield Metric(
                str(qid),
                self,
                self._compute_query(
                    tuple(qrels[qrels["query_id"] == qid][group_col]),
                    tuple(ranking[group_col]),
                    groups,
                )
            )

    def __str__(self):
        name = self.NAME
        cutoff = ""
        if self._cutoff_param is not None:
            cutoff = f"@{self._cutoff_param}"
        group_col = None
        if self._group_col_param != "group":
            group_col = repr(self._group_col_param)
        groups = None
        if self._groups_param is not None:
            groups = repr(",".join(self._groups_param))
        protected_group = None
        if isinstance(self._protected_group_param, ProtectedGroupStrategy):
            if self._protected_group_param != ProtectedGroupStrategy.MINORITY:
                protected_group = repr(self._protected_group_param.value)
        elif isinstance(self._protected_group_param, Hashable):
            protected_group = repr(self._protected_group_param)
        tie_breaking = None
        if isinstance(self._tie_breaking_param, TieBreakingStrategy):
            tie_breaking = repr(self._tie_breaking_param.value)
        elif isinstance(self._tie_breaking_param, Sequence):
            tie_breaking = repr(",".join(self._tie_breaking_param))
        params = [
            f"{name}={param}"
            for name, param in {
                "group_col": group_col,
                "groups": groups,
                "protected_group": protected_group,
                "tie_breaking": tie_breaking,
            }.items()
            if param is not None
        ]
        return f"{name}{cutoff}({','.join(params)})"


class _NormalizedDiscountedDifference(FairnessMeasure):
    NAME = "rND"
    __name__ = "rND"

    def unfairness(
            self,
            ranking: tuple[Hashable],
            groups: set[Hashable],
            group_counts: dict[Hashable, int],
            protected_group: Hashable,
    ) -> float:
        N = sum(group_counts.values())
        S_plus = group_counts[protected_group]

        metric = 0

        # For each ranking position
        for i in range(1, len(ranking)):
            ranking_i = ranking[:i]
            group_counts_i = self.group_counts(ranking_i, groups)

            S_Plus_i = group_counts_i[protected_group]

            metric += (
                    (1 / log(i + 1, 2)) *
                    abs(
                        abs(S_Plus_i / (i + 1)) -
                        abs(S_plus / N)
                    )
            )

        return metric


NormalizedDiscountedDifference = _NormalizedDiscountedDifference()
rND = NormalizedDiscountedDifference
register_measure(rND)


def _kl_divergence(x1: Sequence[float], x2: Sequence[float]) -> float:
    return sum(rel_entr(x1, x2))


class _NormalizedDiscountedKlDivergence(FairnessMeasure):
    SUPPORTED_PARAMS = {
        **FairnessMeasure.SUPPORTED_PARAMS,
        "correct_extreme": ParamInfo(
            dtype=bool,
            required=False,
            default=True,
            desc="correct extreme probability distributions such "
                 "that 0 > P(x) > 1 and 0 > Q(x) > 1"
        ),
    }
    NAME = "rKL"
    __name__ = "rKL"

    @cached_property
    def _correct_extreme(self) -> bool:
        if "correct_extreme" not in self.params:
            return True
        return self.params["correct_extreme"]

    def unfairness(
            self,
            ranking: tuple[Hashable],
            groups: set[Hashable],
            group_counts: dict[Hashable, int],
            protected_group: Hashable,
    ) -> float:
        N = sum(group_counts.values())
        S_Plus = group_counts[protected_group]
        S_Minus = N - S_Plus
        Q = (S_Plus / N, S_Minus / N)
        if self._correct_extreme:
            if S_Plus == N:
                Q_plus = nextafter(Q[0], 0)
                Q = (Q_plus, 1 - Q_plus)
            elif S_Minus == N:
                Q_minus = nextafter(Q[1], 0)
                Q = (1 - Q_minus, Q_minus)

        metric = 0

        # For each ranking position
        for i in range(1, len(ranking)):
            ranking_i = ranking[:i]
            group_counts_i = self.group_counts(ranking_i, groups)

            # P Calculation
            S_Plus_i = group_counts_i[protected_group]
            S_Minus_i = i - S_Plus_i

            P = (S_Plus_i / i, S_Minus_i / i)
            if self._correct_extreme:
                if S_Plus_i == i:
                    P_plus = nextafter(P[0], 0)
                    P = (P_plus, 1 - P_plus)
                elif S_Minus_i == i:
                    P_minus = nextafter(P[1], 0)
                    P = (1 - P_minus, P_minus)

            metric += _kl_divergence(P, Q) / log(i + 1, 2)

        return metric


NormalizedDiscountedKlDivergence = _NormalizedDiscountedKlDivergence()
rKL = NormalizedDiscountedKlDivergence
register_measure(rKL)


class _NormalizedDiscountedRatio(FairnessMeasure):
    NAME = "rRD"
    __name__ = "rRD"

    def unfairness(
            self,
            ranking: tuple[Hashable],
            groups: set[Hashable],
            group_counts: dict[Hashable, int],
            protected_group: Hashable,
    ) -> float:
        N = sum(group_counts.values())
        S_Plus = group_counts[protected_group]
        S_Minus = N - S_Plus
        S_frac: float
        if S_Plus == 0 or S_Minus == 0:
            S_frac = 0
        else:
            S_frac = abs(S_Plus / S_Minus)

        metric = 0

        # For each ranking position
        for i in range(1, len(ranking)):
            ranking_i = ranking[:i]
            group_counts_i = self.group_counts(ranking_i, groups)

            S_Plus_i = group_counts_i[protected_group]
            S_Minus_i = i - S_Plus_i
            S_i_frac: float
            if S_Plus_i == 0 or S_Minus_i == 0:
                S_i_frac = 0
            else:
                S_i_frac = abs(S_Plus_i / S_Minus_i)

            metric += (
                    (1 / log(i + 1, 2)) *
                    abs(S_i_frac - S_frac)
            )

        return metric


NormalizedDiscountedRatio = _NormalizedDiscountedRatio()
rRD = NormalizedDiscountedRatio
register_measure(rRD)


@dataclass
class FairnessEvaluator(Evaluator):
    measures: Final[Iterable[FairnessMeasure]] = field()
    qrels: Final[DataFrame] = field()

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


class FairnessProvider(Provider):
    NAME = "fairness"
    SUPPORTED_MEASURES = [rND, rKL, rRD]
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


_provider = FairnessProvider()
register_provider(_provider)
DefaultPipeline.providers.append(_provider)
