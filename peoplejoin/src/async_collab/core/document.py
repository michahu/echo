import json
import re
from dataclasses import dataclass, field

import json
from rank_bm25 import BM25Okapi

from src.logging_config import general_logger


@dataclass(frozen=True, eq=True, unsafe_hash=True)
class Document:
    """
    A shared document.
    """

    url: str = field(
        metadata={
            "description": "The web address of the document. Should be globally unique."
        }
    )
    title: str = field(metadata={"description": "The title of the document."})
    content: str = field(
        metadata={"description": "The content of the document."}, default=""
    )

    def __str__(self) -> str:
        return self.get_content(trim_content=False)

    def get_content(self, trim_content: bool = False) -> str:
        ret = "Document:\n"
        doc_collection_name = self.url.split("/")[0].strip()
        if len(doc_collection_name) > 0 and (
            not doc_collection_name.startswith("file:")
        ):
            ret += f"Collection name: {doc_collection_name.replace('_',' ')}\n"
        ret += f"Title: {self.title}\n"
        try:
            document_json = json.loads(self.content)
            assert type(document_json) == list, "Content is not a list"
            for i, row in enumerate(document_json):
                ret += f"Record {i+1}: {row}\n"
                if trim_content and i == 2 and len(document_json) > 3:
                    ret += f"... and {len(document_json)-3} more records\n\n"
                    break
        except json.JSONDecodeError:
            ret += f"Content: {self.content}"
        except AssertionError as e:
            general_logger.info(
                f"[Document] Exception in Document.get_content: {e}. Defaulting to raw content"
            )
            ret += f"Content: {self.content}\n"
        return ret

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(**data)


@dataclass
class DocumentCollection:
    documents: list[Document]
    search_index: BM25Okapi | None

    def __init__(self, documents: list[Document]):
        self.documents = documents
        # self.search_index = {doc.title: doc for doc in documents}
        self.search_index = self._build_index(documents)

    def _build_index(self, documents: list[Document]):
        if len(documents) == 0:
            return None
        tokenized_corpus = [self._tokenize(doc.content) for doc in documents]
        return BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> list[str]:
        # A simple tokenizer to split on non-alphanumeric characters
        return re.findall(r"\w+", text.lower())

    def search_documents(self, query: str, top_n: int = 5):
        tokenized_query = self._tokenize(query)
        if self.search_index is None:
            general_logger.info(
                "[DocumentCollection] [WARNING] No documents to search from. Returning empty list."
            )
            return self.documents[:top_n]
        scores = self.search_index.get_scores(tokenized_query)  # type: ignore
        top_n_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_n]
        return [self.documents[i] for i in top_n_indices]
