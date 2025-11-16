# src/ingest_rag.py

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import yaml

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

EMBED_MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")


def split_front_matter(text: str):
    if not text.lstrip().startswith("---"):
        return {}, text

    matches = list(re.finditer(r"^---\s*$", text, flags=re.MULTILINE))
    if len(matches) < 2:
        return {}, text

    start = matches[0].end()
    end = matches[1].start()

    front_matter_str = text[start:end].strip()
    body = text[matches[1].end():].lstrip()

    try:
        meta = yaml.safe_load(front_matter_str) or {}
    except Exception:
        meta = {}

    return meta, body


def load_md_docs(docs_dir: Path):
    headers_to_split_on = [
        ("#", "section"),
        ("##", "subsection"),
        ("###", "subsubsection"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    all_docs = []

    for path in docs_dir.glob("*.md"):
        raw_text = path.read_text(encoding="utf-8")
        base_meta, body = split_front_matter(raw_text)
        md_docs = splitter.split_text(body)

        for d in md_docs:
            chunk_meta = {
                **base_meta,
                **d.metadata,
                "source_file": path.name,
            }
            all_docs.append(
                Document(page_content=d.page_content, metadata=chunk_meta)
            )

    return all_docs


def build_vectorstore_for_hospital(hospital_code: str):
    docs_dir = BASE_DIR / "data" / hospital_code / "docs"
    vs_dir = BASE_DIR / "data" / hospital_code / "vectorstore"
    vs_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔹 院區：{hospital_code}")
    print(f"🔹 讀取 Markdown 文件 ({docs_dir}/*.md)...")
    docs = load_md_docs(docs_dir)
    print(f"👉 共切出 {len(docs)} 個文字區塊 (chunks)")

    if not docs:
        print("⚠️ 沒有讀到任何文件，請先把 .md 放到該院區的 docs/ 資料夾")
        return

    texts = [d.page_content for d in docs]

    print(f"🔹 載入 embedding model：{EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("🔹 計算向量...")
    embeddings = model.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, str(vs_dir / "faiss.index"))
    np.save(vs_dir / "docs.npy", np.array(docs, dtype=object))

    print("✅ 向量庫建立完成！")
    print(f"   儲存路徑：{vs_dir}")


if __name__ == "__main__":
    # 用法：python -m src.ingest_rag wanfang
    if len(sys.argv) < 2:
        print("請輸入院區代碼，例如：python -m src.ingest_rag wanfang")
        sys.exit(1)

    hospital_code = sys.argv[1]
    build_vectorstore_for_hospital(hospital_code)
