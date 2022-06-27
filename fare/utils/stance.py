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


def stance_label(value: float, threshold: float) -> str:
    if isnan(value) or isnan(threshold):
        return "NO"
    elif abs(value) < threshold:
        return "NEUTRAL"
    elif value > 0:
        return "FIRST"
    else:
        return "SECOND"
