import os
from typing import List

import pdfplumber
from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader, TextLoader
from langchain_core.documents import Document

from src.utils.logger import setup_logger
from config import settings

logger = setup_logger(__name__)

# Pages shorter than MIN_PAGE_LENGTH after cleaning are skipped (footers, empty pages).
MIN_PAGE_LENGTH = settings.min_page_length

# Lines containing these patterns are treated as noise (page numbers, headers, footers).
_NOISE_PATTERNS = (
    "page 1",
    "page 2",
    "page 3",
    "page 4",
    "page 5",
)


# Private Helpers

def _is_noise_line(line: str) -> bool:
    """
    Check if a line is a footer, header, or page-number artifact.
    - Empty lines are skipped.
    - Lines that are only a digit are skipped.
    - Lines matching _NOISE_PATTERNS are skipped.
    """
    stripped = line.strip().lower()

    if not stripped:
        return True

    if stripped.isdigit():
        return True

    if any(pattern in stripped for pattern in _NOISE_PATTERNS):
        return True

    return False


def _extract_tables_as_text(page: pdfplumber.page.Page) -> str:
    """
    Extract all tables from a PDF page and convert to readable text.
    Format: 'Header1: Value1 | Header2: Value2'
    Returns empty string if no tables found.
    """
    tables = page.extract_tables()
    if not tables:
        return ""

    table_lines = []

    for table in tables:
        # Need at least a header row and one data row
        if not table or len(table) < 2:
            continue

        headers = [str(cell).strip() if cell else "" for cell in table[0]]

        for row in table[1:]:
            row_parts = [
                f"{header}: {str(cell).strip()}"
                for header, cell in zip(headers, row)
                if cell and str(cell).strip()
            ]
            if row_parts:
                table_lines.append(" | ".join(row_parts))

    return "\n".join(table_lines)


def _extract_plain_text(page: pdfplumber.page.Page) -> str:
    """
    Extract plain text from a PDF page with noise lines removed.
    Used only for pages that contain no tables.
    """
    plain_text = page.extract_text()
    if not plain_text or not plain_text.strip():
        return ""

    filtered_lines = [
        line.strip()
        for line in plain_text.split("\n")
        if not _is_noise_line(line)
    ]

    return "\n".join(filtered_lines).strip()


# PDF Loader

def load_pdf_with_pdfplumber(file_path: str) -> List[Document]:
    """
    Load a PDF file using pdfplumber with structured table extraction.

    Per-page strategy:
        Page HAS tables  -> structured 'Key: Value' text only, plain text skipped.
        Page has NO tables -> cleaned plain text only, no table extraction.

    Pages shorter than MIN_PAGE_LENGTH after processing are skipped.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    documents = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):

                table_text = _extract_tables_as_text(page)

                if table_text:
                    # Table page — use structured text only
                    full_text = table_text.strip()
                    logger.debug("Page %d: table extraction (%d chars)", page_num + 1, len(full_text))
                else:
                    # Non-table page — use plain text only
                    full_text = _extract_plain_text(page)
                    logger.debug("Page %d: plain text extraction (%d chars)", page_num + 1, len(full_text))

                # Skip pages that are too short to be meaningful
                if len(full_text) < MIN_PAGE_LENGTH:
                    logger.warning(
                        "Page %d skipped — too short after cleaning (%d chars)",
                        page_num + 1,
                        len(full_text),
                    )
                    continue

                documents.append(
                    Document(
                        page_content=full_text,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1,
                        },
                    )
                )

    except Exception as exc:
        logger.error("Failed to load PDF '%s': %s", file_path, exc)
        raise RuntimeError(f"Failed to load PDF '{file_path}': {exc}") from exc

    logger.info("PDF '%s' loaded: %d page(s)", file_path, len(documents))
    return documents


# Main Loader

def load_faq_documents(data_folder: str = "data/") -> List[Document]:
    """
    Load all documents from the data folder — supports TXT, PDF, and HTML.
    Walks entire directory tree so nested sub-folders are also scanned.
    """
    if not os.path.exists(data_folder):
        logger.error("Data folder not found: %s", data_folder)
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    try:
        # Load TXT files
        logger.info("Loading TXT files")
        txt_loader = DirectoryLoader(
            data_folder,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        txt_documents = txt_loader.load()
        logger.info("TXT files loaded: %d", len(txt_documents))

        # Load PDF files using pdfplumber
        logger.info("Loading PDF files")
        pdf_documents: List[Document] = []

        for root, _dirs, files in os.walk(data_folder):
            for file_name in files:
                if file_name.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file_name)
                    logger.info("Processing PDF: %s", pdf_path)
                    pdf_documents.extend(load_pdf_with_pdfplumber(pdf_path))

        logger.info("PDF pages loaded: %d", len(pdf_documents))

        # Load HTML files
        logger.info("Loading HTML files")
        html_loader = DirectoryLoader(
            data_folder,
            glob="**/*.html",
            loader_cls=BSHTMLLoader,
        )
        html_documents = html_loader.load()

        # Strip leading/trailing whitespace from HTML content
        for doc in html_documents:
            doc.page_content = doc.page_content.strip()

        logger.info("HTML files loaded: %d", len(html_documents))

        # Combine all documents
        all_documents = txt_documents + pdf_documents + html_documents
        logger.info("Total documents loaded: %d", len(all_documents))
        return all_documents

    except Exception as exc:
        logger.error("Error loading documents: %s", exc)
        raise