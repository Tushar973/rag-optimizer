import streamlit as st
import pandas as pd
import time
import os
from dotenv import load_dotenv
import plotly.express as px

from core.ingestion import load_embedding_model, process_document, create_vector_store
from core.generator import get_llm, build_rag_chain
from core.evaluator import get_judge_llm, evaluate_response

# --------------------------------------------------
# Config
# --------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="RAG Strategy Optimizer",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ RAG Strategy Optimizer")
st.caption(
    "Benchmark how chunk size, overlap, and retrieval depth impact RAG accuracy "
    "using LLM-as-a-Judge evaluation."
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("🧠 RAG System Status")

    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key:
        st.success("LLM Backend: Connected (Groq)")
    else:
        groq_api_key = st.text_input("Enter Groq API Key", type="password")
        if groq_api_key:
            st.success("LLM Backend: Connected (Groq)")
        else:
            st.warning("LLM Backend: Not Connected")

    st.info("Embedding Model: Loaded at Runtime")
    st.info("Vector Store: Built Per Strategy")
    st.info("Evaluation: LLM-as-a-Judge")

    st.markdown("### ⚙️ Active Capabilities")
    st.markdown("""
    - 🔍 Semantic Retrieval  
    - 🧩 Adaptive Chunking  
    - 📊 Strategy Benchmarking  
    - 🧠 LLM-based Evaluation  
    """)

    st.divider()

    st.header("📄 Document Input")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    st.divider()

    st.header("🧪 Experiment Setup")
    test_question = st.text_input(
        "Evaluation Question",
        placeholder="Ask a question that requires deep context..."
    )

    st.markdown("### 🔧 Strategy Parameters")

    with st.expander("⚡ Quick Glance"):
        q_chunk = st.slider("Chunk Size", 256, 1024, 512, 128)
        q_overlap = st.slider("Overlap", 0, 200, 50, 10)
        q_k = st.slider("Top-K", 1, 5, 2)

    with st.expander("⚖️ Balanced"):
        b_chunk = st.slider("Chunk Size ", 512, 2048, 1024, 128)
        b_overlap = st.slider("Overlap ", 0, 400, 200, 20)
        b_k = st.slider("Top-K ", 1, 7, 3)

    with st.expander("🔎 Deep Dive"):
        d_chunk = st.slider("Chunk Size  ", 1024, 4096, 2048, 256)
        d_overlap = st.slider("Overlap  ", 0, 600, 400, 20)
        d_k = st.slider("Top-K  ", 1, 10, 5)

    run_button = st.button("🚀 Run Experiments", use_container_width=True)

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
if run_button and uploaded_file and groq_api_key and test_question:

    strategies = [
        {"name": "Quick Glance", "chunk": q_chunk, "overlap": q_overlap, "k": q_k},
        {"name": "Balanced", "chunk": b_chunk, "overlap": b_overlap, "k": b_k},
        {"name": "Deep Dive", "chunk": d_chunk, "overlap": d_overlap, "k": d_k},
    ]

    st.subheader("⚙️ Running RAG Experiments")

    progress = st.progress(0)
    status = st.empty()

    embeddings = load_embedding_model()
    results = []

    for i, strat in enumerate(strategies):
        status.info(f"Running **{strat['name']}** strategy...")
        start = time.time()

        splits, _ = process_document(
            uploaded_file,
            strat["chunk"],
            strat["overlap"]
        )

        vectorstore = create_vector_store(splits, embeddings)

        llm = get_llm(groq_api_key)
        rag_chain, retriever = build_rag_chain(
            vectorstore,
            llm,
            strat["k"]
        )

        retrieved_docs = retriever.invoke(test_question)
        context_text = "\n".join(d.page_content for d in retrieved_docs)

        answer = rag_chain.invoke(test_question)

        judge_llm = get_judge_llm(groq_api_key)
        metrics = evaluate_response(
            test_question,
            context_text,
            answer,
            judge_llm
        )

        elapsed = round(time.time() - start, 2)

        results.append({
            "Strategy": strat["name"],
            "Chunk Size": strat["chunk"],
            "Overlap": strat["overlap"],
            "Top-K": strat["k"],
            "Response Time (s)": elapsed,
            "Relevance": metrics["relevance_score"],
            "Accuracy": metrics["accuracy_score"],
            "Answer": answer,
            "Judge Reasoning": metrics["reasoning"]
        })

        progress.progress((i + 1) / len(strategies))

    status.success("✅ All experiments completed")

    df = pd.DataFrame(results)

    # --------------------------------------------------
    # Dashboard
    # --------------------------------------------------
    st.subheader("📊 Strategy Comparison")

    col1, col2 = st.columns(2)

    with col1:
        metric_df = df.melt(
            id_vars=["Strategy"],
            value_vars=["Relevance", "Accuracy"],
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            metric_df,
            x="Strategy",
            y="Score",
            color="Metric",
            barmode="group",
            title="Relevance vs Accuracy"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df,
            x="Response Time (s)",
            y="Accuracy",
            size="Relevance",
            color="Strategy",
            title="Latency vs Accuracy Trade-off"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --------------------------------------------------
    # Leaderboard (NO TEXT TRUNCATION)
    # --------------------------------------------------
    st.subheader("🏆 Strategy Leaderboard")

    leaderboard_df = df[[
        "Strategy",
        "Accuracy",
        "Relevance",
        "Response Time (s)",
        "Chunk Size",
        "Top-K"
    ]].sort_values("Accuracy", ascending=False)

    st.dataframe(leaderboard_df, use_container_width=True)

    best = leaderboard_df.iloc[0]
    st.success(
        f"🥇 Best Strategy: **{best['Strategy']}** "
        f"(Accuracy: {best['Accuracy']}/5 | Relevance: {best['Relevance']}/5)"
    )

    # --------------------------------------------------
    # Full Answers (Expandable)
    # --------------------------------------------------
    st.subheader("📝 Detailed Strategy Analysis")

    for _, row in df.iterrows():
        with st.expander(f"📌 {row['Strategy']} | Accuracy: {row['Accuracy']}/5"):
            st.markdown("### ✅ Generated Answer")
            st.write(row["Answer"])

            st.markdown("### 🧠 Judge Reasoning")
            st.write(row["Judge Reasoning"])
