"""PDF text extraction and page-aware chunk preparation for KERNELGPT."""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from pypdf import PdfReader
from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_PDF_SIZE_MB

@dataclass(frozen=True)
class PageText:
    """Extracted text and one-based page number from a PDF."""
    page: int
    text: str

@dataclass(frozen=True)
class TextChunk:
    """Passage with metadata needed for citations and stable indexing."""
    text: str
    source: str
    page: int
    chunk_id: str
    document_id: str
    location: str = ""

def clean_text(text: str) -> str:
    """Normalize whitespace while retaining paragraph boundaries."""
    text = text.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def extract_pdf_pages(file_obj: str | Path | BinaryIO, filename: str | None = None) -> tuple[list[PageText], list[int]]:
    """Extract pages, returning (page text, one-based empty page numbers).

    Image-only pages are reported as empty; extracting them requires future OCR.
    """
    if isinstance(file_obj, (str, Path)):
        path = Path(file_obj)
        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.stat().st_size > MAX_PDF_SIZE_MB * 1024 * 1024:
            raise ValueError(f"PDF exceeds the {MAX_PDF_SIZE_MB} MB size limit.")
        source = filename or path.name
    else:
        source = filename or getattr(file_obj, "name", "uploaded.pdf")
        if not source.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")
        size = getattr(file_obj, "size", None)
        if size is not None and size > MAX_PDF_SIZE_MB * 1024 * 1024:
            raise ValueError(f"PDF exceeds the {MAX_PDF_SIZE_MB} MB size limit.")
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
    try:
        reader = PdfReader(file_obj, strict=False)
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise ValueError("PDF is encrypted and needs a password.")
            except Exception as exc:
                raise ValueError("PDF is encrypted and cannot be opened without a password.") from exc
        pages: list[PageText] = []
        empty_pages: list[int] = []
        for number, pdf_page in enumerate(reader.pages, start=1):
            text = clean_text(pdf_page.extract_text() or "")
            pages.append(PageText(number, text))
            if not text:
                empty_pages.append(number)
        if not pages:
            raise ValueError("PDF contains no pages.")
        if not any(page.text for page in pages):
            raise ValueError("PDF has no extractable text; it may be scanned and require OCR.")
        return pages, empty_pages
    except (ValueError, FileNotFoundError):
        raise
    except Exception as exc:
        raise ValueError(f"Could not read PDF '{source}': {exc}") from exc

def chunk_pages(pages: list[PageText], source: str, *, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[TextChunk]:
    """Split page text into bounded overlapping chunks with page citations."""
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    content_key = source + "\n" + "\n".join(page.text for page in pages)
    document_id = hashlib.sha256(content_key.encode("utf-8")).hexdigest()[:20]
    chunks: list[TextChunk] = []
    for page in pages:
        text = clean_text(page.text)
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            excerpt = text[start:end].strip()
            if excerpt:
                key = f"{document_id}:{page.page}:{start}:{excerpt}"
                chunk_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
                chunks.append(TextChunk(excerpt, source, page.page, chunk_id, document_id))
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks

def process_pdf(file_obj: str | Path | BinaryIO, filename: str | None = None) -> tuple[list[TextChunk], list[int]]:
    """Extract and chunk one PDF; return chunks and pages needing OCR review."""
    pages, empty_pages = extract_pdf_pages(file_obj, filename)
    source = filename or (Path(file_obj).name if isinstance(file_obj, (str, Path)) else getattr(file_obj, "name", "uploaded.pdf"))
    return chunk_pages(pages, source), empty_pages


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    """Create local embeddings in batches with the configured Ollama model."""
    from ollama import Client, RequestError, ResponseError
    from config import EMBEDDING_MODEL, OLLAMA_HOST

    client = Client(host=OLLAMA_HOST)
    vectors: list[list[float]] = []
    try:
        for offset in range(0, len(texts), 100):
            response = client.embed(model=EMBEDDING_MODEL, input=texts[offset:offset + 100], keep_alive="10m")
            vectors.extend([list(vector) for vector in response.embeddings])
    except (RequestError, ResponseError) as exc:
        raise RuntimeError(
            f"Ollama could not create embeddings. Confirm Ollama is running and "
            f"the '{EMBEDDING_MODEL}' model is installed. Details: {exc}"
        ) from exc
    return vectors


def get_collection(persist_dir: str | Path | None = None):
    """Open KERNELGPT's persistent ChromaDB collection."""
    import chromadb
    from config import CHROMA_DIR

    location = Path(persist_dir) if persist_dir else CHROMA_DIR
    location.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(location))
    return client.get_or_create_collection(name="kernelgpt_os_cosine", metadata={"description": "Operating Systems PDF passages", "hnsw:space": "cosine"})


def index_chunks(chunks: list[TextChunk], *, collection=None, embedding_fn=None) -> int:
    """Embed and persist chunks. Injectable collection/embedder supports offline checks."""
    if not chunks:
        return 0
    target = collection or get_collection()
    embed = embedding_fn or _ollama_embed
    vectors = embed([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("Embedding service returned a different number of vectors than input chunks.")
    target.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=vectors,
        metadatas=[{
            "source": chunk.source,
            "page": chunk.page,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "location": chunk.location or f"page {chunk.page}",
        } for chunk in chunks],
    )
    return len(chunks)


def ingest_pdf(file_obj: str | Path | BinaryIO, filename: str | None = None, *, collection=None, embedding_fn=None) -> tuple[int, list[int]]:
    """Extract, chunk, embed, and index a PDF; return count and empty page numbers."""
    chunks, empty_pages = process_pdf(file_obj, filename)
    count = index_chunks(chunks, collection=collection, embedding_fn=embedding_fn)
    return count, empty_pages


def indexed_documents(collection=None) -> list[dict[str, object]]:
    """Return a compact inventory of documents currently in the collection."""
    target = collection or get_collection()
    records = target.get(include=["metadatas"])
    documents: dict[str, dict[str, object]] = {}
    for metadata in records.get("metadatas") or []:
        if not metadata:
            continue
        document_id = str(metadata.get("document_id", ""))
        item = documents.setdefault(document_id, {
            "document_id": document_id,
            "source": metadata.get("source", "unknown"),
            "chunks": 0,
        })
        item["chunks"] = int(item["chunks"]) + 1
    return sorted(documents.values(), key=lambda item: str(item["source"]).lower())
