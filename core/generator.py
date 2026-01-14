from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from flashrank import Ranker, RerankRequest

# 1. Load the lightweight reranker (only loads once)
reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank")

def get_llm(api_key, model_name="llama-3.1-8b-instant"):
    return ChatGroq(groq_api_key=api_key, model_name=model_name)

def build_rag_chain(vectorstore, llm, k=3, use_reranker=True):
    # OPTIMIZATION: Retrieve more docs initially (k*4), then filter down
    initial_k = k * 4 if use_reranker else k
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Custom Reranking Function
    def rerank_docs_func(input_data):
        question = input_data["question"]
        docs = input_data["docs"]
        
        if not use_reranker:
            return docs[:k]
        
        # Convert LangChain docs to FlashRank format
        passages = [
            {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
            for i, doc in enumerate(docs)
        ]
        
        rerank_request = RerankRequest(query=question, passages=passages)
        results = reranker.rerank(rerank_request)
        
        # Sort and take top K
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:k]
        
        # Convert back to simple text for the LLM
        return [res['text'] for res in sorted_results]

    def format_docs_func(docs):
        # Docs are already a list of strings from rerank_docs_func
        return "\n\n".join(docs)
    
    # WRAPPER FIX: Wrap python functions as Runnables
    rerank_chain = RunnableLambda(rerank_docs_func)
    format_chain = RunnableLambda(format_docs_func)
    
    # The Chain Logic
    rag_chain = (
        {"docs": retriever, "question": RunnablePassthrough()} 
        | rerank_chain
        | format_chain 
        | (lambda context: {"context": context, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever