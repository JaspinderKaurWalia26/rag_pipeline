import os
import uuid
from typing import Dict, List

import chromadb
import numpy as np
from langchain_core.documents import Document

from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class VectorStore:
    """Handles storing document embeddings using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "data/vector_store",
    ) -> None:
        """
        Initialize VectorStore with ChromaDB.
        Args:
            collection_name  : Name of the ChromaDB collection.
            persist_directory: Path where ChromaDB persists data to disk.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self) -> None:
        """
        Initialize ChromaDB client and get-or-create the collection.
        Raises:
            RuntimeError: If ChromaDB initialization fails.
        """
        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for RAG"},
            )

            logger.info(
                "Vector store initialized. Collection: %s", self.collection_name
            )
            logger.info(
                "Existing documents in collection: %d", self.collection.count()
            )

        except Exception as exc:
            logger.error("Error initializing vector store: %s", exc)
            raise RuntimeError(f"Error initializing vector store: {exc}") from exc

    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add documents and their embeddings to ChromaDB.
        Args:
            documents : List of LangChain Document objects.
            embeddings: Numpy array of shape (n_docs, embedding_dim).
        Raises:
            ValueError  : If documents/embeddings are empty or counts mismatch.
            RuntimeError: If ChromaDB insertion fails.
        """
        if not documents:
            logger.warning("Empty documents list provided")
            raise ValueError("Documents list cannot be empty")

        if len(documents) != len(embeddings):
            logger.error(
                "Mismatch — documents: %d, embeddings: %d",
                len(documents), len(embeddings),
            )
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        logger.info("Adding %d documents to vector store...", len(documents))

        try:
            ids: List[str] = []
            metadatas: List[Dict] = []
            documents_text: List[str] = []
            embeddings_list: List[List[float]] = []

            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(doc_id)

                metadata = dict(doc.metadata)
                metadata["doc_index"] = i
                metadata["content_length"] = len(doc.page_content)
                metadatas.append(metadata)

                documents_text.append(doc.page_content)
                embeddings_list.append(embedding.tolist())

            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )

            logger.info(
                "Added %d documents. Total in collection: %d",
                len(documents),
                self.collection.count(),
            )

        except Exception as exc:
            logger.error("Error adding documents to vector store: %s", exc)
            raise RuntimeError(
                f"Error adding documents to vector store: {exc}"
            ) from exc

    def count(self) -> int:
        """
        Return the number of documents currently in the collection.
        Returns:
            Integer count of stored documents.
        """
        return self.collection.count()
