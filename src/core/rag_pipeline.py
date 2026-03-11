from langchain_ollama import ChatOllama
from src.core.retriever import ChromaRetriever
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# System Prompt
SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions ONLY using the provided context.

RULES:
1. Use ONLY information from the context below. Do NOT use general knowledge.
2. If the answer is not in the context, say: "I don't know. This information is not available in the provided documents."
3. Keep answers concise and accurate.
4. Do not hallucinate facts, numbers, or details not present in the context.
5. For questions about yourself, say: "I am an AI assistant. I don't have personal goals or identity."
"""


async def rag_with_system_prompt(
    query: str,
    retriever: ChromaRetriever,
    llm: ChatOllama,
    top_k: int = 2
) -> str:
    """
    Core async RAG pipeline — retrieves context and generates answer via LLM.
    Args:
        query:    User question
        retriever: ChromaRetriever instance
        llm:      ChatOllama instance
        top_k:    Number of documents to retrieve
    Returns:
        Answer string from LLM
    Raises:
        RuntimeError: If LLM call fails
    """

    # Retrieve relevant context chunks from ChromaDB
    docs = retriever.retrieve(query, top_k)

    if not docs:
        logger.warning("No relevant context found for query.")
        return "No relevant context found."

    # Build the prompt by injecting retrieved context
    context = "\n\n".join(d["content"] for d in docs)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    try:
        # Call LLM asynchronously 
        logger.info("Generating answer from LLM...")
        response = await llm.ainvoke(prompt)
        logger.info("Answer generated successfully")
        return response.content

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise RuntimeError(f"LLM call failed: {e}")


def create_rag_chain(retriever: ChromaRetriever, llm: ChatOllama, top_k: int = 2):
    """
    Returns a reusable async RAG chain function.
    Args:
        retriever: ChromaRetriever instance
        llm:       ChatOllama instance
        top_k:     Number of documents to retrieve
    Returns:
        Async RAG chain coroutine function
    """
    logger.info("Creating async RAG chain...")

    
    async def rag_chain(question: str, k: int = top_k) -> str:
        return await rag_with_system_prompt(question, retriever, llm, top_k=k)

    return rag_chain