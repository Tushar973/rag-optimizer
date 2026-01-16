from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from flashrank import Ranker, RerankRequest


# --------------------------------------------------
# 1. Load reranker once (lightweight & fast)
# --------------------------------------------------
reranker = Ranker(
    model_name="ms-marco-TinyBERT-L-2-v2",
    cache_dir="/tmp/flashrank"
)

# --------------------------------------------------
# 2. LLM Loader
# --------------------------------------------------
def get_llm(api_key, model_name="llama-3.1-8b-instant"):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.2,   # 🔥 more factual, less hallucination
        max_tokens=700
    )

# --------------------------------------------------
# 3. RAG Chain Builder
# --------------------------------------------------
def build_rag_chain(vectorstore, llm, k=3, use_reranker=True):
    """
    RAG pipeline with optional reranking.
    Logic preserved; answer quality improved via prompt engineering.
    """

    # Retrieve more docs first, rerank later
    initial_k = k * 4 if use_reranker else k
    retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})

    # --------------------------------------------------
    # 🔥 Improved Prompt (MAIN UPGRADE)
    # --------------------------------------------------
    template = """
You are an expert technical assistant.

Answer the question using ONLY the information provided in the context.
Do NOT use prior knowledge.

Guidelines:
- Be precise and complete
- Preserve mathematical expressions and equations if present
- Explain concepts step-by-step when reasoning is required
- If the context is insufficient, clearly say so
- Do not guess or hallucinate

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)

    # --------------------------------------------------
    # 4. Reranking Logic (UNCHANGED)
    # --------------------------------------------------
    def rerank_docs_func(input_data):
        question = input_data["question"]
        docs = input_data["docs"]

        if not use_reranker:
            return docs[:k]

        passages = [
            {
                "id": str(i),
                "text": doc.page_content,
                "meta": doc.metadata
            }
            for i, doc in enumerate(docs)
        ]

        rerank_request = RerankRequest(
            query=question,
            passages=passages
        )

        results = reranker.rerank(rerank_request)

        top_k = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )[:k]

        return [res["text"] for res in top_k]

    # --------------------------------------------------
    # 5. Context Formatting (Improved)
    # --------------------------------------------------
    def format_docs_func(docs):
        """
        Formats retrieved chunks into a clean, readable context block.
        """
        formatted = []
        for i, doc in enumerate(docs):
            formatted.append(f"[Chunk {i+1}]\n{doc}")
        return "\n\n".join(formatted)

    rerank_chain = RunnableLambda(rerank_docs_func)
    format_chain = RunnableLambda(format_docs_func)

    # --------------------------------------------------
    # 6. RAG Chain (STRUCTURE UNCHANGED)
    # --------------------------------------------------
    rag_chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough()
        }
        | rerank_chain
        | format_chain
        | (lambda context: {
            "context": context,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
