import os
import time

import requests
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from rag.embed_store import VectorStore
from rag.generate import generate_answer
from rag.ingest import build_chunk_records, load_documents

BASE_DATA_DIR = "data"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 512
OVERLAP = 48

os.environ["LOCAL_LLM_CHUNK_LIMIT"] = "2"

def download_model():
    SentenceTransformer(EMBEDDING_MODEL).save("model/SentenceTransformer")
    CrossEncoder(RERANKING_MODEL).save("model/CrossEncoder")


def ingest(dataset_folder: str):
    docs = load_documents(dataset_folder)
    chunks = build_chunk_records(docs=docs, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    return docs, chunks

local_llm_chunk_limit = os.getenv("LOCAL_LLM_CHUNK_LIMIT")
online = os.getenv("ONLINE")

if torch.cuda.is_available():
    os.environ["DEVICE"] = "cuda"
elif torch.xpu.is_available():
    os.environ["DEVICE"] = "xpu"
elif torch.backends.mps.is_available():
    os.environ["DEVICE"] = "mps"
else:
    os.environ["DEVICE"] = "cpu"

try:
    requests.get("https://python.org", timeout=10)
    os.environ["ONLINE"] = "true"
except (requests.ConnectTimeout, requests.ConnectionError, requests.ReadTimeout):
    os.environ["ONLINE"] = "false"

if not os.path.isfile("model/SentenceTransformer/model.safetensors") or not os.path.isfile("model/CrossEncoder/model.safetensors"):
    download_model()


st.set_page_config(page_title="Course Notes RAG", page_icon="", layout="wide")

@st.cache_resource(show_spinner="Updating model...")
def update_model():
    if online == "true":
        download_model()

update_model()

datasets = sorted([
    f for f in os.listdir(BASE_DATA_DIR)
    if os.path.isdir(os.path.join(BASE_DATA_DIR, f))
])

if len(datasets) == 0:
    datasets.append(BASE_DATA_DIR)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = None
if "ingested_datasets" not in st.session_state:
    st.session_state.ingested_datasets = {}
if "last_load_time" not in st.session_state:
    st.session_state.last_load_time = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None

store = st.session_state.vector_store

sb = st.sidebar
with sb:
    st.subheader("About")
    st.markdown(
        f"- Embedding Model: `{EMBEDDING_MODEL}` (local)\n"
        f"- Re-ranking Model: `{RERANKING_MODEL}` (local)\n"
        f"- Chunking Strategy: `TikToken` token-aware text splitter\n"
        f"- Chunk Size and Overlap: (`{CHUNK_SIZE}`, `{OVERLAP}`)\n"
        f"- Local Model Processor: `{os.getenv('DEVICE').upper()}`\n"
        f"- Internet Connection: `{'TRUE' if online == 'true' else 'FALSE'}`"
    )

st.title("RAG-Based AI Search System for Course Notes")
st.caption("Select a course and ask a question below.")

col_dataset, col_search = st.columns([2, 8], vertical_alignment="bottom")

selected_dataset = col_dataset.selectbox(
    label="Course",
    label_visibility="collapsed",
    options=datasets,
    index=0
)

with col_search.form("search_form", clear_on_submit=False, border=False):
    col_input, col_button = st.columns([8, 2], vertical_alignment="bottom")
    query = col_input.text_input(
        "Question",
        placeholder="Type your question here.",
        label_visibility="collapsed",
    )

    search_clicked = col_button.form_submit_button("Search", type="primary", width="stretch")

if st.session_state.selected_dataset != selected_dataset:
    load_start = time.perf_counter()
    if selected_dataset in st.session_state.ingested_datasets:
        with st.spinner("Switching dataset..."):
            store.switch_collection(selected_dataset)

    else:
        with st.spinner("Loading and indexing dataset..."):
            dataset_folder = os.path.join(BASE_DATA_DIR, selected_dataset)
            docs, chunks = ingest(dataset_folder)

            store.build(chunks, selected_dataset)

            st.session_state.ingested_datasets[selected_dataset] = {
                "docs": docs,
                "chunks": chunks
            }
            st.session_state.selected_dataset = selected_dataset

    load_end = time.perf_counter()
    load_time = load_end - load_start

    st.session_state.selected_dataset = selected_dataset
    st.session_state.last_load_time = load_time

active_dataset = st.session_state.ingested_datasets.get(st.session_state.selected_dataset, {"docs": [], "chunks": []})
docs = active_dataset["docs"]
chunks = active_dataset["chunks"]


with sb:
    st.divider()
    st.subheader("Settings")
    mode_display = {"llm": "LLM", "extractive": "Extractive"}
    mode = st.selectbox(
        label="Answer Mode",
        options=["llm", "extractive"],
        index=0,
        format_func=lambda x: mode_display.get(x),
        help="Extractive works with no setup. LLM mode needs a provider setup done correctly.",
    )
    provider_display = {
        "google": "Google (Gemini Flash Lite)",
        "local": "Local (Phi Mini) [Experimental]",
    }
    provider = st.selectbox(
        label="LLM Provider",
        options=["google", "local"],
        index=0,
        format_func=lambda x: provider_display.get(x),
        disabled=False if mode == "llm" else True,
        help=f"Requires provider API key set in environment (except local provider).\nLocal is experimental, has chunk read limited to {local_llm_chunk_limit}, and is very prone to hallucination; use as LLM fallback only.",
    )
    max_chunks = len(chunks)
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=int(local_llm_chunk_limit) if provider == "local" else (10 if max_chunks >= 10 else (max_chunks if max_chunks > 0 else 2)),
        value=int(local_llm_chunk_limit) if provider == "local" else (3 if max_chunks >= 3 else (max_chunks if max_chunks > 0 else 1)),
    )
    score_threshold = st.slider(
        "Chunk score threshold to retrieve",
        min_value=0.00,
        max_value=1.00,
        value=0.20
    )
    st.divider()
    st.header("Dataset")
    st.write(f"`{selected_dataset}` loaded in **{st.session_state.last_load_time:.2f}** seconds.")
    st.caption(f"Indexed **{len(docs)}** documents \u2192 **{len(chunks)}** chunks")
    with st.expander("Documents in this index"):
        doc_list = ""
        for d in docs:
            doc_list = (
                doc_list
                + '- "'
                + d["title"]
                + '" by '
                + d["author"]
                + " ("
                + d["date"]
                + ")"
                + "\n"
            )
        st.markdown(doc_list)

if search_clicked and query.strip():
    if chunks:
        with st.spinner("Generating answer..."):
            answer_start = time.perf_counter()
            retrieved = store.query(query, top_k=top_k, threshold=score_threshold)
            answer, meta = generate_answer(query, retrieved, mode=mode, provider=provider, dataset=selected_dataset)
            answer_end = time.perf_counter()
            answer_latency = answer_end - answer_start

            st.session_state.search_results = {
                "retrieved": retrieved,
                "answer": answer,
                "latency": answer_latency,
                "mode": meta["mode"],
                "provider": meta["provider"],
                "provider_raw": meta["provider_raw"]
            }
    else:
        st.warning("No doument loaded.")

elif search_clicked:
    st.warning("Type a question first.")

if st.session_state.search_results:
    res = st.session_state.search_results
    if res["provider_raw"] == "local":
        res["retrieved"] = res["retrieved"][:int(local_llm_chunk_limit)]
    st.subheader("Answer")
    st.caption(f"{res["mode"]} response generated in **{res["latency"]:.2f} seconds**{f" using **{res["provider"]}**" if res["mode"] == "LLM" else ""}.")
    st.write(res["answer"])

    if res["retrieved"]:
        st.divider()
        st.subheader("Sources")
        for chunk, score in res["retrieved"]:
            with st.container(border=True):
                if score >= 0.75:
                    score_color = "green"
                elif score >= 0.5:
                    score_color = "yellow"
                elif score >= 0.25:
                    score_color = "orange"
                else:
                    score_color = "red"
                st.markdown(
                    f"**{chunk.chunk_id}** :{score_color}-badge[Score: {score:.2f}]"
                )
                st.caption(f"{chunk.doc_author} · {chunk.doc_date}")

                with st.expander("View Excerpt"):
                    st.write(chunk.text)
