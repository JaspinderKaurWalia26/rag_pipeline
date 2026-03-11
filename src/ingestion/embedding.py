from sentence_transformers import SentenceTransformer
from src.utils.logger import setup_logger
from typing import List
import numpy as np

logger = setup_logger(__name__)


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize EmbeddingManager with a SentenceTransformer model.
        Args:
            model_name: Name of the SentenceTransformer model to load
        """
        # Store model name
        self.model_name = model_name

        # Placeholder for the loaded model
        self.model = None

        # Load the embedding model
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the SentenceTransformer model.
        Raises:
            RuntimeError: If model fails to load
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        Args:
            texts: List of strings to embed
        Returns:
            numpy array of embeddings
        Raises:
            ValueError: If model not loaded or texts is empty
        """

        # Check if model is loaded
        if not self.model:
            logger.error("Embedding model is not loaded")
            raise ValueError("Embedding model is not loaded")

        # Check if texts list is empty
        if not texts:
            logger.warning("Empty texts list provided")
            raise ValueError("Texts list cannot be empty")

        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts, show_progress_bar=False)
            logger.info("Embeddings generated successfully")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise