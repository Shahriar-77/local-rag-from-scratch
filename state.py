
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from core.ingestion import ingest_file
from core.chunking import chunk_document
from core.retrieval import embed_chunks
from generation import MODEL_CONFIGS


DATA_DIR = Path("data_state")
DATA_DIR.mkdir(exist_ok=True)

CORPUS_PATH = DATA_DIR / "corpus.pkl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"


_corpus: Optional[List[Dict]] = None
_embeddings: Optional[np.ndarray] = None
_embedding_model: Optional[SentenceTransformer] = None
_llm_cache: Dict[str, Llama] = {}
_last_answer: Optional[str] = None


def load_state():
    global _corpus, _embeddings

    if CORPUS_PATH.exists():
        with open(CORPUS_PATH, "rb") as f:
            _corpus = pickle.load(f)

    if EMBEDDINGS_PATH.exists():
        _embeddings = np.load(EMBEDDINGS_PATH)

    return _corpus, _embeddings


def save_state(corpus, embeddings):
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(corpus, f)
    np.save(EMBEDDINGS_PATH, embeddings)


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def build_or_update_corpus(paths: List[Path], rebuild=False):
    global _corpus, _embeddings

    if rebuild or _corpus is None:
        _corpus = []
        _embeddings = None

    existing_sources = {c["source"] for c in _corpus} if _corpus else set()
    new_chunks = []

    for path in paths:
        if str(path) in existing_sources:
            continue
        doc = ingest_file(str(path))
        new_chunks.extend(chunk_document(doc))

    if not new_chunks:
        return _corpus

    model = get_embedding_model()
    new_embeddings = model.encode([c["text"] for c in new_chunks])

    _embeddings = (
        new_embeddings if _embeddings is None
        else np.vstack([_embeddings, new_embeddings])
    )

    _corpus.extend(new_chunks)
    save_state(_corpus, _embeddings)
    return _corpus


def get_corpus():
    global _corpus
    if _corpus is None:
        load_state()
    return _corpus


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        load_state()
    return _embeddings


def get_llm(model_name: str) -> Llama:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name not in _llm_cache:
        cfg = MODEL_CONFIGS[model_name]
        _llm_cache[model_name] = Llama(
            model_path=cfg["path"],
            n_ctx=cfg["n_ctx"],
            n_gpu_layers=cfg["default_gpu_layers"],
            verbose=False
        )

    return _llm_cache[model_name]


def get_last_answer():
    return _last_answer


def set_last_answer(answer: str):
    global _last_answer
    _last_answer = answer
