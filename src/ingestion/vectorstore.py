"""
Vector store: embed German legal text using multilingual model,
store in Qdrant with rich metadata for filtered retrieval.
"""

import logging
import uuid
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.ingestion.loader import LegalDocument, load_all_laws

logger = logging.getLogger(__name__)

# Multilingual model — handles German legal text natively
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
COLLECTION_NAME = "arbeitsrecht_de"
QDRANT_PATH = "./data/qdrant_local"  # local on-disk for dev; swap to URL for prod


def get_embeddings():
    """Load multilingual sentence transformer."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def build_vector_store(documents: list[LegalDocument]) -> QdrantVectorStore:
    """
    Chunk legal paragraphs, embed with multilingual model,
    and upsert into Qdrant with source metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n(", "\n", ". ", " "],
    )

    texts, metadatas = [], []
    for doc in documents:
        chunks = splitter.split_text(doc.text)
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "source": doc.source,
                "paragraph": doc.paragraph,
                "title": doc.title,
                "url": doc.url,
                "doc_type": doc.doc_type,
                "id": str(uuid.uuid4()),
            })

    logger.info(f"Embedding {len(texts)} chunks from {len(documents)} paragraphs…")

    embeddings = get_embeddings()
    client = get_qdrant_client()

    # Create collection if not exists
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    logger.info(f"✓ Indexed {len(texts)} chunks into Qdrant collection '{COLLECTION_NAME}'")
    return vector_store


def load_vector_store() -> QdrantVectorStore:
    """Load existing Qdrant collection (after indexing)."""
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )


if __name__ == "__main__":
    docs = load_all_laws()
    build_vector_store(docs)
