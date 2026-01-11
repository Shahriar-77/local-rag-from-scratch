


import hashlib


def make_chunk_id(source_path: str, local_id: str) -> str:
    raw = f"{source_path}::{local_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def chunk_pdf(doc: dict) -> list:
    chunks = []

    for block in doc["blocks"]:
        page_idx = block["page_index"]
        text = block["text"].strip()

        if not text:
            continue

        chunk_id = make_chunk_id(
            doc["source_path"],
            f"page_{page_idx}"
        )

        chunks.append({
            "id": chunk_id,
            "source": doc["source_path"],
            "file_type": "pdf",
            "text": text,
            "metadata": {
                "page_index": page_idx,
                "block_type": block["type"]
            }
        })

    return chunks

def chunk_docx(doc: dict) -> list:
    chunks = []
    current_heading = None
    buffer = []

    for block in doc["blocks"]:
        if block["type"] == "heading":
            if buffer:
                chunk_id = make_chunk_id(
                    doc["source_path"],
                    current_heading or "intro"
                )

                chunks.append({
                    "id": chunk_id,
                    "source": doc["source_path"],
                    "file_type": "docx",
                    "text": "\n".join(buffer),
                    "metadata": {
                        "section": current_heading
                    }
                })

                buffer = []

            current_heading = block["text"]

        elif block["type"] == "paragraph":
            buffer.append(block["text"])

    # flush last section
    if buffer:
        chunk_id = make_chunk_id(
            doc["source_path"],
            current_heading or "final"
        )

        chunks.append({
            "id": chunk_id,
            "source": doc["source_path"],
            "file_type": "docx",
            "text": "\n".join(buffer),
            "metadata": {
                "section": current_heading
            }
        })

    return chunks

def chunk_pptx(doc: dict) -> list:
    chunks = []

    for slide in doc["blocks"]:
        slide_idx = slide["slide_index"]

        texts = []
        title = None

        for el in slide["elements"]:
            if el["role"] == "title":
                title = el["text"]
            texts.append(el["text"])

        content = "\n".join(texts).strip()
        if not content:
            continue

        chunk_id = make_chunk_id(
            doc["source_path"],
            f"slide_{slide_idx}"
        )

        chunks.append({
            "id": chunk_id,
            "source": doc["source_path"],
            "file_type": "pptx",
            "text": content,
            "metadata": {
                "slide_index": slide_idx,
                "title": title
            }
        })

    return chunks

def chunk_csv(doc: dict) -> list:
    chunk_id = make_chunk_id(
        doc["source_path"],
        "table"
    )

    return [{
        "id": chunk_id,
        "source": doc["source_path"],
        "file_type": "csv",
        "text": doc["text"],
        "metadata": {
            "schema": doc["metadata"]["schema"],
            "num_rows": doc["metadata"]["num_rows"]
        }
    }]


def chunk_txt(doc: dict, max_chars: int = 800, overlap: int = 100) -> list:
    text = doc["text"]
    chunks = []

    start = 0
    idx = 0

    while start < len(text):
        end = start + max_chars
        chunk_text = text[start:end].strip()

        if not chunk_text:
            break

        chunk_id = make_chunk_id(
            doc["source_path"],
            f"chunk_{idx}"
        )

        chunks.append({
            "id": chunk_id,
            "source": doc["source_path"],
            "file_type": "txt",
            "text": chunk_text,
            "metadata": {
                "char_start": start,
                "char_end": min(end, len(text))
            }
        })

        start = end - overlap
        idx += 1

    return chunks


def chunk_xlsx(doc: dict) -> list:
    chunks = []

    for block in doc["blocks"]:
        sheet_name = block["sheet_name"]
        df = block["dataframe"]

        chunk_id = make_chunk_id(
            doc["source_path"],
            f"sheet_{sheet_name}"
        )

        chunks.append({
            "id": chunk_id,
            "source": doc["source_path"],
            "file_type": "xlsx",
            "text": df.to_csv(index=False),
            "metadata": {
                "sheet_name": sheet_name,
                "num_rows": df.shape[0],
                "num_columns": df.shape[1],
                "columns": list(df.columns)
            }
        })

    return chunks


def chunk_document(doc: dict) -> list:
    file_type = doc["file_type"]

    if file_type == "pdf":
        return chunk_pdf(doc)
    elif file_type == "docx":
        return chunk_docx(doc)
    elif file_type == "pptx":
        return chunk_pptx(doc)
    elif file_type == "csv":
        return chunk_csv(doc)
    elif file_type =='txt':
        return chunk_txt(doc)
    elif file_type == 'xlsx':
        return chunk_xlsx(doc)
    else:
        raise ValueError(f"No chunker for {file_type}")

