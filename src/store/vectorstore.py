import os
import uuid
import numpy as np
import chromadb
from langchain_core.documents import Document
from src.utils.logger import setup_logger
from typing import List

logger = setup_logger(__name__)


class VectorStore:
    """Handles storing document embeddings in ChromaDB."""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "data/vector_store"
    ):
        """
        Initialize VectorStore with ChromaDB.
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Path to persist ChromaDB data
        """
        # Store collection details
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # ChromaDB client and collection
        self.client = None
        self.collection = None

        # Initialize vector store
        self._initialize_store()

    def _initialize_store(self) -> None:
        """
        Initialize ChromaDB client and collection.
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Ensure persistence directory exists
            os.makedirs(self.persist_directory, exist_ok=True)

            # Create persistent ChromaDB client
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for RAG"}
            )

            logger.info(f"Vector store initialized. Collection: {self.collection_name}")
            logger.info(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise RuntimeError(f"Error initializing vector store: {e}")

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """
        Add documents and their embeddings to ChromaDB.
        Args:
            documents: List of LangChain Document objects
            embeddings: Numpy array of embeddings
        Raises:
            ValueError: If documents or embeddings are empty or mismatched
        """

        # Check if documents list is empty
        if not documents:
            logger.warning("Empty documents list provided")
            raise ValueError("Documents list cannot be empty")

        # Validate input sizes match
        if len(documents) != len(embeddings):
            logger.error("Documents and embeddings count mismatch")
            raise ValueError("Number of documents must match number of embeddings")

        logger.info(f"Adding {len(documents)} documents to vector store...")

        try:
            ids = []
            metadatas = []
            documents_text = []
            embeddings_list = []

            # Prepare data for ChromaDB
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):

                # Generate unique document ID
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(doc_id)

                # Prepare metadata
                metadata = dict(doc.metadata)
                metadata["doc_index"] = i
                metadata["content_length"] = len(doc.page_content)
                metadatas.append(metadata)

                # Store document content and embedding
                documents_text.append(doc.page_content)
                embeddings_list.append(embedding.tolist())

            # Add data to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )

            logger.info(f"Added {len(documents)} documents. Total now: {self.collection.count()}")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise