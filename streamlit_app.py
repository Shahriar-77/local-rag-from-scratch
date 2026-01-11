
import streamlit as st
from app import answer_query, collect_files
from state import build_or_update_corpus, get_corpus


st.set_page_config(page_title="Local Hybrid RAG", layout="wide")
st.title("Local Hybrid RAG (Lexical + Semantic)")


st.sidebar.header("Corpus")
data_dir = st.sidebar.text_input("Data directory", "data/raw")
rebuild = st.sidebar.checkbox("Rebuild corpus")

if st.sidebar.button("Ingest"):
    files = collect_files([data_dir])
    build_or_update_corpus(files, rebuild=rebuild)
    st.sidebar.success("Corpus ready")

corpus = get_corpus()
if corpus:
    st.sidebar.write(f"Chunks: {len(corpus)}")


query = st.text_input("Query")

col1, col2 = st.columns(2)
with col1:
    model = st.selectbox("Model", ["mistral7b","phi3"])
with col2:
    top_k = st.slider("Top-K", 1, 10, 3)

if st.button("Run") and query:
    with st.spinner("Running hybrid retrieval..."):
        answer, tokens = answer_query(query, model=model, top_k=top_k)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Token Usage")
    st.json(tokens)
