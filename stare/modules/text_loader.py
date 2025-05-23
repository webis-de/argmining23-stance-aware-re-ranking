from dataclasses import dataclass
from json import loads

from pandas import DataFrame
from pyterrier.transformer import Transformer
from tqdm.auto import tqdm

from stare.config import CONFIG


@dataclass
class TextLoader(Transformer):
    verbose: bool = False

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        document_ids: set[str] = set(topics_or_res["docno"].tolist())
        document_texts: dict[str, str] = {}
        with CONFIG.corpus_file_path.open("r") as lines:
            if self.verbose:
                num_lines = sum(1 for _ in lines)
                lines.seek(0)
                lines = tqdm(
                    lines,
                    total=num_lines,
                    desc="Scanning corpus"
                )
            for line in lines:
                document_dict = loads(line)
                document_id = document_dict["id"]
                if document_id in document_ids:
                    document_texts[document_id] = document_dict["contents"]
        topics_or_res["text"] = topics_or_res["docno"].map(
            lambda document_id: document_texts[document_id]
        )
        return topics_or_res
