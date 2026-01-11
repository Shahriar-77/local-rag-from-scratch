
from typing import Literal
from llama_cpp import Llama

MODEL_CONFIGS = {
    "phi3": {
        "path": "models/Phi-3-mini-4k-instruct-q4.gguf",
        "n_ctx": 4096, # 2048
        "default_gpu_layers": 14
    },
    "mistral7b": {
        "path": "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "n_ctx": 6000, # 2048
        "default_gpu_layers": 14
    }
}

def run_llm_inference(
    prompt: str,
    model_name: Literal["phi3", "mistral7b"] = "phi3",
    max_tokens: int = 256,
    temperature: float = 0.2,
    system_prompt: str = None,
) -> str:
    """
    Run inference using a selected lightweight LLM.

    Parameters
    ----------
    prompt : str
        The user question or instruction to answer.
    model_name : {'phi3', 'mistral7b'}
        - 'phi3': Phi-3-mini-4k-instruct (lighter, faster)
        - 'mistral7b': Mistral-7B-Instruct-v0.2-Q4_K_M (larger, stronger)
    max_tokens : int
        Upper bound on generation length. Keep moderate to avoid OOM.
    temperature : float
        Sampling temperature; lower is more deterministic.
    system_prompt : str | None
        Instructions that govern assistant behavior.

    Returns
    -------
    str
        The assistant’s generated text.
    """

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Choose 'phi3' or 'mistral7b'."
        )

    config = MODEL_CONFIGS[model_name]

    # Default system prompt if not provided
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. "
            "Use context responsibly and answer the query clearly."
        )

    llm = Llama(
        model_path=config["path"],
        n_gpu_layers=config["default_gpu_layers"],
        n_ctx=config["n_ctx"],
        verbose=False
    )

    # Use chat API for both instruct models
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    
    )

    return response["choices"][0]["message"]["content"]


def assemble_context(results, max_tokens: int = 3000) -> str:
    """
    Assemble retrieved chunks into a single context string for LLM inference.

    This function concatenates the top-ranked retrieved chunks while enforcing
    an approximate token budget. The default limit (3000 tokens) is chosen to
    accommodate small-to-mid sized instruction-tuned LLMs running on
    consumer-grade GPUs (e.g., RTX 3060 with ~6GB VRAM).

    Why this constraint exists:
    - Larger context windows increase VRAM usage and inference latency.
    - Quantized 7B-class models typically operate safely within a 2k–4k token window.
    - Exceeding this budget may cause OOM errors or severe slowdowns.

    Advanced users with more VRAM (or CPU-based inference) may safely increase
    this value, provided their selected model supports larger context windows.

    Parameters
    ----------
    results : list of dict
        Ranked retrieval results. Each entry must contain:
        - "text": chunk text
        - "source": source document identifier
        - "id": stable chunk identifier

    max_tokens : int, optional (default=3000)
        Approximate token budget for the assembled context.

    Returns
    -------
    str
        A single formatted context string suitable for RAG-style prompting.
    """

    context_blocks = []
    token_count = 0

    for chunk in results:
        text = chunk["text"]
        source = chunk.get("source", "unknown")
        chunk_id = chunk.get("id", "unknown")

        # Approximate token count (1 token ≈ 0.75 words is model-dependent)
        approx_tokens = len(text.split())

        if token_count + approx_tokens > max_tokens:
            break

        block = (
            f"[Source: {source} | Chunk ID: {chunk_id}]\n"
            f"{text}\n"
        )

        context_blocks.append(block)
        token_count += approx_tokens

    return "\n".join(context_blocks)

def generate_llm_answer(
    context: str,
    query: str,
    model: str,
    system_prompt: str = (
        "You are a technical assistant.\n"
        "Answer the question using ONLY the provided context.\n"
        "If the answer cannot be found in the context, say so explicitly."
    ),
) -> str:
    """
    Generate an answer using a lightweight, VRAM-constrained LLM for RAG.

    This function performs grounded generation by combining:
    - a system prompt (behavioral constraint),
    - retrieved context (knowledge grounding),
    - and a user query.

    The model must be explicitly selected from the supported options to ensure
    predictable VRAM usage and inference behavior.

    Supported models:
    - "mistral7b": higher reasoning quality, ~4–5GB VRAM (4-bit)
    - "phi3": lower VRAM, faster inference, weaker reasoning
    - defaults to "phi3"

    Parameters
    ----------
    context : str
        Assembled retrieval context produced by `assemble_context`.

    query : str
        User query to be answered.

    model : str
        Model identifier. Must be one of:
        - "mistral7b"
        - "phi3"

    system_prompt : str, optional
        Instruction to guide model behavior. Defaults to a strict
        context-grounded RAG prompt.

    Returns
    -------
    str
        Generated answer text.
    """

    if model not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model '{model}'. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )


    if not context.strip():
        return (
        "No relevant information was found in the retrieved documents "
        "to answer this question."
    )


    prompt = f"""
{system_prompt}

Context:
---------
{context}

Question:
---------
{query}

Answer:
""".strip()

    
    # Inference backend is currently llama.cpp via llama_cpp.
    # This function can be extended to support alternative backends
    # (e.g., vLLM, Triton, remote APIs) without changing call sites.


    if len(prompt) > 12000:
        raise ValueError("Prompt too long for configured context window.")

    answer = run_llm_inference(
        prompt=prompt,
        model_name=model,
    )

    return answer
