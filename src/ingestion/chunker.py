from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.logger import setup_logger
from config import settings
from typing import List

logger = setup_logger(__name__)


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks.
    Args:
        documents: List of LangChain Document objects
    Returns:
        List of chunked Document objects
    Raises:
        ValueError: If documents list is empty
    """

    # Check if documents list is empty
    if not documents:
        logger.warning("Empty documents list provided")
        raise ValueError("Documents list cannot be empty")

    logger.info(f"Starting to chunk {len(documents)} documents")

    try:
        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

        all_chunks = []

        # Process each document
        for doc_idx, doc in enumerate(documents):

            # Extract content and metadata from LangChain Document
            content = doc.page_content
            metadata = doc.metadata

            # Skip empty documents
            if not content.strip():
                logger.warning(f"Document {doc_idx + 1} is empty, skipping...")
                continue

            # Split document into chunks
            chunks = splitter.split_text(content)
            logger.info(f"Document {doc_idx + 1}/{len(documents)}: {len(chunks)} chunks created")

            # Store valid chunks with metadata
            for idx, chunk_text in enumerate(chunks):

                # Skip very short chunks
                if len(chunk_text.strip()) < 15:
                    continue

                all_chunks.append(Document(
                    page_content=chunk_text.strip(),
                    metadata={
                        **metadata,
                        "chunk_number": idx,
                        "chunk_total": len(chunks)
                    }
                ))

        logger.info(f"Chunking completed. Total chunks created: {len(all_chunks)}")
        return all_chunks

    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        raise