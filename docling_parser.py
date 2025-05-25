from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseBlobParser
from langchain_core.document_loaders.blob_loaders import Blob
from typing import Iterator, Dict, Any

class DoclingStreamBlobParser(BaseBlobParser):
    """Parser using Docling's DocumentConverter for binary streams."""
    def __init__(self, document_name: str):
        self.document_name = document_name
        self.converter = DocumentConverter()

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        with blob.as_bytes_io() as f:
            docling_source = DocumentStream(name=self.document_name, stream=f)
            result = self.converter.convert(docling_source)
            docling_doc = result.document
            content = docling_doc.export_to_markdown()
            metadata: Dict[str, Any] = getattr(docling_doc, "metadata", {}) or {}
            metadata["source"] = self.document_name
            yield Document(page_content=content, metadata=metadata)
