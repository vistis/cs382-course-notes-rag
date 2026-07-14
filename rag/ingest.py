import os
import time
from dataclasses import dataclass
from typing import List
from pypdf import PdfReader

from langchain_text_splitters import TokenTextSplitter


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    doc_author: str
    doc_date: str
    text: str


def load_documents(folder: str) -> List[dict]:
    """Load every .txt, .pdf, .md file in `folder` into dicts."""
    docs = []
    valid_extensions = (".txt", ".pdf", ".md")
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(valid_extensions):
            continue

        path = os.path.join(folder, filename)

        # Default metadata
        text = ""
        title = os.path.splitext(filename)[0].replace("_", " ") #.title()
        # author = pwd.getpwuid(os.stat(path).st_uid).pw_name
        author = "Unknown Author"
        date = time.strftime("%Y-%b-%d", (time.strptime(time.ctime(os.path.getctime(path)))))

        # Load .txt and .md
        if filename.endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()

        # Load .pdf
        elif filename.endswith(".pdf"):
            reader = PdfReader(path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            text.strip()

            metadata = reader.metadata
            # Prefer to use the title from filename - metadata is usually misleading
            # if metadata.title:
            #     title = metadata.title
            if metadata.author:
                author = metadata.author
            if metadata.creation_date:
               date = metadata.creation_date.strftime("%Y-%b-%d")

        if not text:
            continue

        docs.append({"text": text, "title": title, "author": author, "date": date})
    return docs


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Split text into overlapping token-aware chunks using OpenAI TikToken."""
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunks = splitter.split_text(text)
    return chunks


def build_chunk_records(
    docs: List[dict], chunk_size: int = 80, overlap: int = 20
) -> List[Chunk]:
    """Turn loaded documents into a flat list of Chunk records ready for embedding."""
    records = []
    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            records.append(
                Chunk(
                    chunk_id=f"{doc['title']}::{i}", doc_title=doc["title"], doc_author=doc["author"], doc_date=doc["date"], text=piece
                )
            )
    return records
