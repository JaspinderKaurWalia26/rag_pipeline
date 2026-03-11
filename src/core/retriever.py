import chromadb
from typing import List, Dict, Optional
from src.ingestion.embedding import EmbeddingManager
from src.utils.logger import setup_logger
from config import settings

logger = setup_logger(__name__)

# Only one instance of EmbeddingManager is created and reused
_embedding_manager: Optional[EmbeddingManager] = None


def _get_embedder() -> EmbeddingManager:
    """
    Get or create a singleton EmbeddingManager instance.
    Returns:
        EmbeddingManager instance
    """
    global _embedding_manager
    if _embedding_manager is None:
        logger.info("Initializing EmbeddingManager")
        _embedding_manager = EmbeddingManager(settings.embedding_model)
    return _embedding_manager


class ChromaRetriever:
    """ChromaDB based document retriever."""

    def __init__(
        self,
        collection_name: str = settings.collection_name,
        persist_directory: str = settings.vector_store_path
    ):
        """
        Initialize ChromaRetriever with persistent ChromaDB connection.
        Args:
            collection_name: ChromaDB collection name
            persist_directory: Path to ChromaDB persistence directory
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client once
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)

            # Get collection - raises error if not found
            try:
                self.collection = self.client.get_collection(name=collection_name)
            except Exception:
                logger.error(f"Collection not found: {collection_name}")
                raise RuntimeError(f"Collection not found: {collection_name}")

            logger.info(f"ChromaRetriever initialized. Collection: {collection_name}")

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise RuntimeError(f"Failed to initialize ChromaDB client: {e}")

    def retrieve(self, query: str, top_k: int = settings.top_k_results) -> List[Dict]:
        """
        Retrieve top-k relevant documents for a query.
        Args:
            query: Search query string
            top_k: Number of results to return
        Returns:
            List of dicts with content, metadata, and similarity score
        Raises:
            ValueError: If query is empty
        """

        # Check if query is empty
        if not query.strip():
            logger.warning("Empty query provided")
            raise ValueError("Query cannot be empty")

        try:
            # Generate embedding for query
            embedder = _get_embedder()
            query_embedding = embedder.generate_embeddings([query])[0].tolist()

            # Perform vector search
            logger.info(f"Searching top {top_k} similar documents...")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Format results with similarity scores
            docs: List[Dict] = []
            for content, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                docs.append({
                    "content": content,
                    "metadata": metadata,
                    "similarity_score": round(1 - distance / 2, 4),
                })

            logger.info(f"Found {len(docs)} similar documents")
            return docs

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise