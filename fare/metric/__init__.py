from ir_measures import Measure, parse_measure as _parse_measure
from ir_measures.measures import registry

from fare.metric.f1 import F1, F1_Touche


def parse_measure(measure: str) -> Measure:
    for additional_measure in (F1(), F1_Touche()):
        registry[additional_measure.NAME] = additional_measure
    return _parse_measure(measure)
