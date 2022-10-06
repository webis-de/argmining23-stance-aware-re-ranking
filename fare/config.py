from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dataclasses_json import dataclass_json, LetterCase, config
from yaml import safe_load

from fare.modules.fairness_reranker import FairnessReranker
from fare.modules.stance_reranker import StanceReranker
from fare.modules.stance_tagger import StanceTagger


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

    stance_tagger: StanceTagger = field(
        metadata=config(
            encoder=lambda tagger: tagger.value,
            decoder=StanceTagger
        )
    )
    stance_filter_threshold: float

    stance_reranker: StanceReranker = field(
        metadata=config(
            encoder=lambda reranker: reranker.value,
            decoder=StanceReranker
        )
    )
    stance_reranker_cutoff: Optional[int]

    fairness_reranker: FairnessReranker = field(
        metadata=config(
            encoder=lambda reranker: reranker.value,
            decoder=FairnessReranker
        )
    )
    fairness_reranker_cutoff: Optional[int]

    metrics: list[str]

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
