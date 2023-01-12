from ir_measures import Measure, parse_measure as _parse_measure

from fare.metric.f1 import F1


def parse_measure(measure: str) -> Measure:
    if measure.split("@")[0] == "F1":
        return F1()
    return _parse_measure(measure)
