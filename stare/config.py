from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from dataclasses_json import LetterCase, config, DataClassJsonMixin
from ir_measures import Measure
from yaml import safe_load

# noinspection PyUnresolvedReferences
import stare.metric.fairness
from stare.metric import parse_measure
from stare.modules.stance_reranker import StanceReranker
from stare.modules.stance_tagger import StanceTagger


@dataclass(frozen=True)
class RunConfig(DataClassJsonMixin):
    dataclass_json_config = config(
        letter_case=LetterCase.CAMEL)["dataclasses_json"]

    stance_tagger: StanceTagger = field(
        metadata=config(
            decoder=StanceTagger
        ),
        default=StanceTagger.ORIGINAL,
    )
    stance_tagger_cutoff: Optional[int] = None
    stance_tagger_threshold: float = 0.0
    stance_randomization_cutoff: Optional[int] = None
    stance_randomization_target_f1: float = 1.0
    stance_reranker: StanceReranker = field(
        metadata=config(
            decoder=StanceReranker
        ),
        default=StanceReranker.ORIGINAL,
    )
    stance_reranker_cutoff: Optional[int] = None


@dataclass(frozen=True)
class Config(DataClassJsonMixin):
    dataclass_json_config = config(
        letter_case=LetterCase.CAMEL)["dataclasses_json"]

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

    max_teams: Optional[int] = None
    max_runs_per_team: Optional[int] = None

    runs: List[RunConfig] = field(default_factory=list)
    stance_tagger_zero_shot_score_threshold: float = 0.0

    measures_relevance: list[Measure] = field(metadata=config(
        decoder=lambda metrics: [parse_measure(metric) for metric in metrics]
    ), default_factory=list)
    measures_quality: list[Measure] = field(metadata=config(
        decoder=lambda metrics: [parse_measure(metric) for metric in metrics]
    ), default_factory=list)
    measures_diversity_relevance: list[Measure] = field(metadata=config(
        decoder=lambda metrics: [parse_measure(metric) for metric in metrics]
    ), default_factory=list)
    measures_diversity_quality: list[Measure] = field(metadata=config(
        decoder=lambda metrics: [parse_measure(metric) for metric in metrics]
    ), default_factory=list)
    measures_stance: list[Measure] = field(metadata=config(
        decoder=lambda metrics: [parse_measure(metric) for metric in metrics]
    ), default_factory=list)

    significance_level = None

    filter_by_qrels: bool = False

    open_ai_api_key: Optional[str] = None

    def __post_init__(self):
        self.cache_directory_path.mkdir(exist_ok=True, parents=True)

    @classmethod
    def load(cls, config_path: Path) -> "Config":
        with config_path.open("r") as config_file:
            config_dict = safe_load(config_file)
            return Config.from_dict(config_dict)


CONFIG: Config = Config.load(Path(__file__).parent.parent / "config.yml")
