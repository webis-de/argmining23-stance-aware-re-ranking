from ir_measures import Measure, parse_measure as _parse_measure
from ir_measures.measures import registry

from stare.metric.classification import (
    F1, NumJudged, FreqFirst, FreqSecond, FreqNeutral, FreqNo
)


def parse_measure(measure: str) -> Measure:
    for additional_measure in (
            F1(), NumJudged(), FreqFirst(), FreqSecond(), FreqNeutral(),
            FreqNo(),
    ):
        registry[additional_measure.NAME] = additional_measure
    return _parse_measure(measure)
