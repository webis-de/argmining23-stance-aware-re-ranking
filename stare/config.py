from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from dataclasses_json import dataclass_json, LetterCase, config
from ir_measures import Measure
from yaml import safe_load

# noinspection PyUnresolvedReferences
import stare.metric.fairness
from stare.metric import parse_measure
from stare.modules.diversity_reranker import DiversityReranker
from stare.modules.effectiveness_reranker import EffectivenessReranker
from stare.modules.fairness_reranker import FairnessReranker
from stare.modules.stance_tagger import StanceTagger


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class RunConfig:
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
    effectiveness_reranker: EffectivenessReranker = field(
        metadata=config(
            decoder=EffectivenessReranker
        ),
        default=EffectivenessReranker.ORIGINAL,
    )
    effectiveness_reranker_cutoff: Optional[int] = None
    diversity_reranker: DiversityReranker = field(
        metadata=config(
            decoder=DiversityReranker
        ),
        default=DiversityReranker.ORIGINAL,
    )
    diversity_reranker_cutoff: Optional[int] = None
    fairness_reranker: FairnessReranker = field(
        metadata=config(
            decoder=FairnessReranker
        ),
        default=FairnessReranker.ORIGINAL,
    )
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

    max_teams: Optional[int] = None
    max_runs_per_team: Optional[int] = None
    team_runs: Optional[set[str]] = field(metadata=config(
        decoder=lambda runs: set(runs) if runs is not None else None
    ), default=None)

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
    measures_per_query: bool = False

    significance_level: Optional[float] = None
    effect_size: bool = False

    filter_by_qrels: bool = False

    offline: bool = False

    open_ai_api_key: Optional[str] = None

    def __post_init__(self):
        self.cache_directory_path.mkdir(exist_ok=True, parents=True)

    @classmethod
    def load(cls, config_path: Path) -> "Config":
        with config_path.open("r") as config_file:
            config_dict = safe_load(config_file)
            return Config.from_dict(config_dict)



CONFIG: Config = Config.load(Path(__file__).parent.parent / "config.yml")
