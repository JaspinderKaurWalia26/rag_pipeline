from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, BSHTMLLoader
from langchain_core.documents import Document
from src.utils.logger import setup_logger
from typing import List
import os

logger = setup_logger(__name__)


def load_faq_documents(data_folder: str = "data/") -> List[Document]:
    """
    Load all documents from the data folder (TXT, PDF, HTML).
    Args:
        data_folder: Path to the data folder
    Returns:
        List of loaded documents
    """

    # Check if data folder exists
    if not os.path.exists(data_folder):
        logger.error(f"Data folder not found: {data_folder}")
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    try:
        # Load TXT files
        logger.info("Loading TXT files")
        txt_loader = DirectoryLoader(
            data_folder,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        txt_documents = txt_loader.load()
        logger.info(f"TXT files loaded: {len(txt_documents)}")

        # Load PDF files
        logger.info("Loading PDF files")
        pdf_loader = DirectoryLoader(
            data_folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        pdf_documents = pdf_loader.load()
        logger.info(f"PDF files loaded: {len(pdf_documents)}")

        # Load HTML files
        logger.info("Loading HTML files")
        html_loader = DirectoryLoader(
            data_folder,
            glob="**/*.html",
            loader_cls=BSHTMLLoader
        )
        html_documents = html_loader.load()
        logger.info(f"HTML files loaded: {len(html_documents)}")

        # Combine all documents
        all_docs = txt_documents + pdf_documents + html_documents
        logger.info(f"Total documents loaded: {len(all_docs)}")

        return all_docs

    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise