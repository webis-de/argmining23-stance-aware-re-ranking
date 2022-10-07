"""
Fairness measures from "Evaluating Fairness in Argument Retrieval"
Paper: https://doi.org/10.1145/3459637.3482099
Code: https://github.com/sachinpc1993/fair-arguments
"""
from math import log

from ir_measures import define_byquery
from pandas import DataFrame
from scipy.special import rel_entr


def _normalized_discounted_difference(ranking_df: DataFrame) -> float:
    pro_count = list(ranking_df["pro_count"].unique())[0]
    con_count = list(ranking_df["con_count"].unique())[0]
    N = pro_count + con_count

    rnd_list = []

    # For each ranking position
    for i in range(1, 6):
        temp_ranking = ranking_df[ranking_df["rank"].isin(range(1, i + 1))]

        stance_freq = temp_ranking["stance"].value_counts()
        stance_freq_df = stance_freq.to_frame().reset_index().rename(columns={
            "stance": "count",
            "index": "stance"
        })

        protected_group = list(temp_ranking["protected_group"].unique())[0]

        if len(stance_freq_df[
                   stance_freq_df["stance"] == protected_group
               ]["count"]) == 0:
            S_Plus_in_i = 0
        else:
            S_Plus_in_i = list(stance_freq_df[
                                   stance_freq_df["stance"] == protected_group
                                   ]["count"])[0]

        if protected_group == "PRO":
            S_plus = pro_count
        else:
            S_plus = con_count

        intermediate_rnd = (
                (1 / log(i + 1, 2)) *
                abs(abs(S_Plus_in_i / (i + 1)) - abs(S_plus / N))
        )

        rnd_list.append(intermediate_rnd)

    final_rnd = sum(rnd_list)

    return final_rnd


rND = define_byquery(
    _normalized_discounted_difference,
    name="rND",
    support_cutoff=True,
)


def _normalized_discounted_kl_divergence(ranking_df: DataFrame) -> float:
    protected_group = list(ranking_df["protected_group"].unique())[0]
    pro_count = list(ranking_df["pro_count"].unique())[0]
    con_count = list(ranking_df["con_count"].unique())[0]
    N = pro_count + con_count

    rKL_list = []

    for i in range(1, 6):
        temp_ranking = ranking_df[ranking_df["rank"].isin(range(1, i + 1))]

        stance_freq = temp_ranking["stance"].value_counts()
        stance_freq_df = stance_freq.to_frame().reset_index().rename(columns={
            "stance": "count",
            "index": "stance"
        })

        protected_group_list_in_i = stance_freq_df[
            stance_freq_df["stance"] == protected_group
            ]["count"]
        group_list_in_i = stance_freq_df[
            stance_freq_df["stance"] != protected_group
            ]["count"]

        # For P Vector Generation
        if len(protected_group_list_in_i) == 0:
            S_Plus_in_i = 0
        else:
            S_Plus_in_i = list(protected_group_list_in_i)[0]

        # For Q Vector Generation
        if len(group_list_in_i) == 0:
            S_Minus_in_i = 0
        else:
            S_Minus_in_i = list(group_list_in_i)[0]

        # P Calculation
        P = [S_Plus_in_i / i, S_Minus_in_i / i]

        # Q Calculation
        if protected_group == "PRO":
            Q = [pro_count / N, con_count / N]
        else:
            Q = [con_count / N, pro_count / N]

        kl_pq = rel_entr(P, Q)
        rKL = sum(kl_pq) / log(i + 1, 2)
        rKL_list.append(rKL)

    final_rKL = sum(rKL_list)

    return final_rKL


rKL = define_byquery(
    _normalized_discounted_kl_divergence,
    name="rKL",
    support_cutoff=True,
)


def _normalized_discounted_ratio(ranking_df: DataFrame) -> float:
    pro_count = list(ranking_df["pro_count"].unique())[0]
    con_count = list(ranking_df["con_count"].unique())[0]

    rRD_list = []

    for i in range(1, 6):
        temp_ranking = ranking_df[ranking_df["rank"].isin(range(1, i + 1))]

        stance_freq = temp_ranking["stance"].value_counts()
        stance_freq_df = stance_freq.to_frame().reset_index().rename(columns={
            "stance": "count",
            "index": "stance"
        })

        protected_group = list(temp_ranking["protected_group"].unique())[0]

        if len(stance_freq_df[
                   stance_freq_df["stance"] == protected_group
               ]["count"]) == 0:
            S_Plus_in_i = 0
        else:
            S_Plus_in_i = list(stance_freq_df[
                                   stance_freq_df["stance"] == protected_group
                                   ]["count"])[0]

        if len(stance_freq_df[
                   stance_freq_df["stance"] != protected_group
               ]["count"]) == 0:
            S_Minus_in_i = 0
        else:
            S_Minus_in_i = list(stance_freq_df[
                                    stance_freq_df["stance"] != protected_group
                                    ]["count"])[0]

        if protected_group == "PRO":
            S_plus = pro_count
            S_Minus = con_count
        else:
            S_plus = con_count
            S_Minus = pro_count

        if (S_Plus_in_i == 0) or (S_Minus_in_i == 0):
            intermediate_rrd = (
                    (1 / log(i + 1, 2)) *
                    abs(0 - abs(S_plus / S_Minus))
            )
        else:
            intermediate_rrd = (
                    (1 / log(i + 1, 2)) *
                    abs(
                        abs(S_Plus_in_i / S_Minus_in_i) -
                        abs(S_plus / S_Minus)
                    )
            )

        rRD_list.append(intermediate_rrd)

    final_rrd = sum(rRD_list)

    return final_rrd


rRD = define_byquery(
    _normalized_discounted_ratio,
    name="rRD",
    support_cutoff=True,
)
