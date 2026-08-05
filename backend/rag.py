import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from langchain_community.document_loaders import DirectoryLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from config import EMBED_MODEL, CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MODE_FOLDERS

embeddings = SentenceTransformerEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"local_files_only": True},
)

def load_documents(folder):
    docs = []

    # Load CSV files — autodetect_encoding tries multiple encodings so files
    # with special characters load instead of being skipped. silent_errors keeps
    # the rebuild going even if one file is truly unreadable.
    csv_loader = DirectoryLoader(
        folder,
        glob="**/*.csv",
        loader_cls=CSVLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
        silent_errors=True,
    )
    docs.extend(csv_loader.load())

    # Load PDF files one by one
    import glob as glob_module
    pdf_files = glob_module.glob(f"{folder}/**/*.pdf", recursive=True)
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())

    # Load JSON intent files (used by the emergency mode: tag/patterns/responses)
    import json
    json_files = glob_module.glob(f"{folder}/**/*.json", recursive=True)
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for intent in data.get("intents", []):
                tag       = intent.get("tag", "")
                patterns  = " ".join(intent.get("patterns", []))
                responses = " ".join(intent.get("responses", []))
                text = f"Topic: {tag}\nQuestions: {patterns}\nFirst Aid: {responses}"
                docs.append(Document(page_content=text, metadata={"source": json_path, "tag": tag}))
        except Exception as e:
            print(f"⚠️  Could not read {json_path}: {e}")

    return docs

def build_vectorstore(mode_name):
    folder = MODE_FOLDERS[mode_name]
    docs   = load_documents(folder)

    if not docs:
        print(f"⚠️  {mode_name} → no documents found in {folder}, skipping")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        print(f"⚠️  {mode_name} → documents were empty after splitting, skipping")
        return

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=mode_name,
        persist_directory=CHROMA_PATH
    )
    vectorstore.persist()
    print(f"✅ {mode_name} → {len(chunks)} chunks saved to ChromaDB")

def get_vectorstore(mode_name):
    return Chroma(
        collection_name=mode_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

# ── HYBRID SEARCH (BM25 keyword + dense vector, fused with RRF) ──
# BM25 catches exact keyword matches (drug names, test names); dense vectors
# catch meaning/synonyms. RRF (Reciprocal Rank Fusion) blends both rankings.
from rank_bm25 import BM25Okapi

_bm25_cache = {}   # mode_name -> (BM25 index, list of Documents)

def _get_bm25(mode_name):
    """Build (once, then cache) a BM25 keyword index from a mode's stored chunks."""
    if mode_name in _bm25_cache:
        return _bm25_cache[mode_name]
    db = get_vectorstore(mode_name)
    stored = db.get()                          # pull all chunks back out of ChromaDB
    texts = stored["documents"]
    from langchain_core.documents import Document
    docs = [Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, stored.get("metadatas", [{}] * len(texts)))]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    _bm25_cache[mode_name] = (bm25, docs)
    return bm25, docs

def _rrf_fuse(dense_docs, bm25_ranked, k=60, top_k=TOP_K):
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across both lists."""
    scores = {}
    for rank, doc in enumerate(dense_docs):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + 1.0 / (k + rank)
    for rank, doc in enumerate(bm25_ranked):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + 1.0 / (k + rank)
    # Map content back to a Document object
    by_content = {d.page_content: d for d in dense_docs + bm25_ranked}
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [by_content[c] for c in ranked[:top_k]]

def search(mode_name, query):
    """Hybrid search: dense (Chroma) + keyword (BM25), fused with RRF."""
    db = get_vectorstore(mode_name)
    dense_docs = db.similarity_search(query, k=TOP_K)

    bm25, docs = _get_bm25(mode_name)
    if bm25 is None or not docs:
        return dense_docs   # fallback if BM25 unavailable

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_idx = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    bm25_docs = [docs[i] for i in ranked_idx]

    return _rrf_fuse(dense_docs, bm25_docs)

if __name__ == "__main__":
    for mode in MODE_FOLDERS.keys():
        build_vectorstore(mode)
    print("\n✅ ALL MODES LOADED INTO CHROMADB!")