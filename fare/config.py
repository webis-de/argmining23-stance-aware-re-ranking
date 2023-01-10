from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from dataclasses_json import dataclass_json, LetterCase, config
from ir_measures import parse_measure, Measure
from yaml import safe_load

# noinspection PyUnresolvedReferences
import fare.metric.fairness
from fare.modules.fairness_reranker import FairnessReranker
from fare.modules.stance_reranker import StanceReranker
from fare.modules.stance_tagger import StanceTagger


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class RunConfig:
    stance_tagger: StanceTagger = field(metadata=config(
        decoder=StanceTagger
    ))
    stance_reranker: StanceReranker = field(metadata=config(
        decoder=StanceReranker
    ))
    fairness_reranker: FairnessReranker = field(
        metadata=config(
            decoder=FairnessReranker
        )
    )
    stance_tagger_threshold: float = 0.0
    stance_tagger_cutoff: Optional[int] = None
    stance_reranker_cutoff: Optional[int] = None
    fairness_reranker_cutoff: Optional[int] = None


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Config:
    topics_file_path: Path = field(
        metadata=config(encoder=str, decoder=Path)
    )
    qrels_relevance_file_path: Path = field(
        metadata=config(encoder=str, decoder=Path)
    )
    qrels_quality_file_path: Path = field(
        metadata=config(encoder=str, decoder=Path)
    )
    qrels_stance_file_path: Path = field(
        metadata=config(encoder=str, decoder=Path)
    )
    runs_directory_path: Path = field(
        metadata=config(encoder=str, decoder=Path)
    )
    corpus_file_path: Path = field(
        metadata=config(encoder=str, decoder=Path)
    )
    cache_directory_path: Path = field(
        metadata=config(encoder=str, decoder=Path)
    )
    metrics_output_file_path: Path = field(
        metadata=config(encoder=str, decoder=Path)
    )

    runs: List[RunConfig]

    measures_relevance: list[Measure] = field(metadata=config(
        decoder=lambda metrics: [parse_measure(metric) for metric in metrics]
    ))
    measures_quality: list[Measure] = field(metadata=config(
        decoder=lambda metrics: [parse_measure(metric) for metric in metrics]
    ))
    measures_stance: list[Measure] = field(metadata=config(
        decoder=lambda metrics: [parse_measure(metric) for metric in metrics]
    ))
    measures_diversity: list[Measure] = field(metadata=config(
        decoder=lambda metrics: [parse_measure(metric) for metric in metrics]
    ))
    measures_per_query: bool

    filter_by_qrels: bool

    offline: bool

    def __post_init__(self):
        self.cache_directory_path.mkdir(exist_ok=True)

    @classmethod
    def load(cls, config_path: Path) -> "Config":
        with config_path.open("r") as config_file:
            config_dict = safe_load(config_file)
            return Config.from_dict(config_dict)


CONFIG: Config = Config.load(Path(__file__).parent.parent / "config.yml")
