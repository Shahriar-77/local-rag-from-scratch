
from pathlib import Path
from typing import List

from state import (
    build_or_update_corpus,
    get_corpus,
    get_embeddings,
    get_last_answer,
    set_last_answer,
    get_llm
)

from core.retrieval import hybrid_search
from generation import assemble_context, generate_llm_answer


def collect_files(paths: List[str]) -> List[Path]:
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(f for f in path.rglob("*") if f.is_file())
        else:
            files.append(path)
    return files


def estimate_tokens(llm, text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8")))


def answer_query(query: str, model="mistral7b", top_k=3):
    corpus = get_corpus()
    embeddings = get_embeddings()

    results = hybrid_search(
        query=query,
        chunks=corpus,
        embeddings=embeddings,
        top_k=top_k
    )

    context = assemble_context(results)

    prev = get_last_answer()
    if prev:
        context = f"[Previous Answer]\n{prev}\n\n{context}"

    answer = generate_llm_answer(
        context=context,
        query=query,
        model=model
    )

    set_last_answer(answer)

    llm = get_llm(model)
    tokens = {
        "query": estimate_tokens(llm, query),
        "context": estimate_tokens(llm, context),
        "answer": estimate_tokens(llm, answer),
    }
    tokens["total"] = sum(tokens.values())

    return answer, tokens
