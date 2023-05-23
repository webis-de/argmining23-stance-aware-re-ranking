from math import nan, isnan


def stance_value(label: str) -> float:
    if label == "FIRST":
        return 1
    elif label == "SECOND":
        return -1
    elif label == "NEUTRAL":
        return 0
    elif label == "NO":
        return nan
    else:
        raise ValueError(f"Unknown stance label: {label}")


def stance_label(value: float) -> str:
    if isnan(value):
        return "NO"
    elif value > 0:
        return "FIRST"
    elif value < 0:
        return "SECOND"
    else:
        return "NEUTRAL"
