from src.ingestion.loaders import load_faq_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedding import EmbeddingManager
from src.store.vectorstore import VectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Load documents
logger.info("Starting data ingestion pipeline")
faq_docs = load_faq_documents("data/")

# Chunk documents 
chunks = chunk_documents(faq_docs)  

# Generate embeddings
embedding_manager = EmbeddingManager("all-MiniLM-L6-v2")
texts = [doc.page_content for doc in chunks]
embeddings = embedding_manager.generate_embeddings(texts)

# Add to VectorStore
vectorstore = VectorStore(collection_name="faq_docs", persist_directory="data/vector_store")
vectorstore.add_documents(chunks, embeddings)

logger.info("Data ingestion pipeline completed!")