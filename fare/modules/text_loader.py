from dataclasses import dataclass
from json import loads

from pandas import DataFrame
from pyterrier.transformer import Transformer
from tqdm.auto import tqdm

from fare.config import CONFIG


@dataclass
class TextLoader(Transformer):

    def transform(self, ranking: DataFrame) -> DataFrame:
        ranking = ranking.copy()
        document_ids: set[str] = set(ranking["docno"].tolist())
        document_texts: dict[str, str] = {}
        with CONFIG.corpus_file_path.open("r") as corpus_file:
            num_lines = sum(1 for _ in corpus_file)
            corpus_file.seek(0)
            lines = tqdm(
                corpus_file,
                total=num_lines,
                desc="Scanning corpus"
            )
            for line in lines:
                document_dict = loads(line)
                document_id = document_dict["id"]
                if document_id in document_ids:
                    document_texts[document_id] = document_dict["contents"]
        ranking["text"] = ranking["docno"].map(
            lambda document_id: document_texts[document_id]
        )
        return ranking
