import streamlit as st
import pandas as pd
import asyncio
from core.ingestion import load_embedding_model, process_document, create_vector_store
from core.generator import get_llm, build_rag_chain
from core.evaluator import get_judge_llm, evaluate_response
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="RAG Optimizer", layout="wide")

st.title("⚡ Efficient RAG Parameter Optimizer")

# Sidebar Configuration
with st.sidebar:
    # Check if key exists in environment
    env_key = os.getenv("GROQ_API_KEY")
    
    if env_key:
        # If found, set the variable and show a success badge
        groq_api_key = env_key
        st.success("✅ API Key loaded securely from .env")
    else:
        # If NOT found, show the input box so you can paste it manually
        groq_api_key = st.text_input("Groq API Key", type="password")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    
    st.markdown("### Experiment Settings")
    test_question = st.text_input("Test Question for Evaluation")
    
    # Define Strategies
    st.write("Defining Parallel Strategies:")
    strategies = [
        {"name": "Quick Glance", "chunk_size": 512, "overlap": 50, "k": 2},
        {"name": "Balanced", "chunk_size": 1024, "overlap": 200, "k": 3},
        {"name": "Deep Dive", "chunk_size": 2048, "overlap": 400, "k": 5}
    ]

# Main Logic
if st.button("Run Experiments") and uploaded_file and groq_api_key and test_question:
    
    embeddings = load_embedding_model()
    results = []
    
    progress_bar = st.progress(0)
    
    for i, strat in enumerate(strategies):
        st.write(f"Running Strategy: **{strat['name']}**...")
        
        # 1. Ingest Data (Specific to strategy)
        splits, _ = process_document(uploaded_file, strat['chunk_size'], strat['overlap'])
        vectorstore = create_vector_store(splits, embeddings)
        
        # 2. Generate Answer
        llm = get_llm(groq_api_key)
        rag_chain, retriever = build_rag_chain(vectorstore, llm, k=strat['k'])
        
        # Get context for the judge
        retrieved_docs = retriever.invoke(test_question)
        context_text = "\n".join([d.page_content for d in retrieved_docs])
        
        # Generate
        answer = rag_chain.invoke(test_question)
        
        # 3. Evaluate (LLM-as-a-Judge)
        judge_llm = get_judge_llm(groq_api_key)
        eval_metrics = evaluate_response(test_question, context_text, answer, judge_llm)
        
        # 4. Record Results
        results.append({
            "Strategy": strat['name'],
            "Chunk Size": strat['chunk_size'],
            "Overlap": strat['overlap'],
            "Response Time": "N/A", # You can add time tracking here
            "Answer": answer,
            "Relevance (1-5)": eval_metrics['relevance_score'],
            "Accuracy (1-5)": eval_metrics['accuracy_score'],
            "Judge Reasoning": eval_metrics['reasoning']
        })
        
        progress_bar.progress((i + 1) / len(strategies))

    # Display Results
    st.success("Experiments Completed!")
    df = pd.DataFrame(results)
    
    # Metric Visualization
    st.subheader("🏆 Performance Leaderboard")
    st.dataframe(df[["Strategy", "Relevance (1-5)", "Accuracy (1-5)", "Chunk Size"]])
    
    # Detailed Analysis
    st.subheader("📝 Detailed Analysis")
    for index, row in df.iterrows():
        with st.expander(f"Strategy: {row['Strategy']} (Score: {row['Relevance (1-5)']}/5)"):
            st.markdown(f"**Answer:** {row['Answer']}")
            st.markdown(f"**Judge's Reasoning:** {row['Judge Reasoning']}")