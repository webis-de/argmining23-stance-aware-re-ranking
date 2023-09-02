from pandas import DataFrame


def reset_order(res: DataFrame) -> DataFrame:
    res["rank"] = res.groupby("qid", sort=False).cumcount() + 1
    res["score"] = -res["rank"]
    return res
