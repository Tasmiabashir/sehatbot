
from langchain_community.document_loaders import DirectoryLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
os.environ["HF_HUB_OFFLINE"] = "1"        # use the locally cached model
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # never phone HuggingFace at startup

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from backend.config import EMBED_MODEL, CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MODE_FOLDERS

embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

def load_documents(folder):
    docs = []

    # Load CSV files
    csv_loader = DirectoryLoader(folder, glob="**/*.csv", loader_cls=CSVLoader)
    docs.extend(csv_loader.load())

    # Load PDF files one by one
    import glob as glob_module
    pdf_files = glob_module.glob(f"{folder}/**/*.pdf", recursive=True)
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())

    return docs

def build_vectorstore(mode_name):
    folder = MODE_FOLDERS[mode_name]
    docs   = load_documents(folder)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

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

def search(mode_name, query):
    db = get_vectorstore(mode_name)
    return db.similarity_search(query, k=TOP_K)

if __name__ == "__main__":
    for mode in MODE_FOLDERS.keys():
        build_vectorstore(mode)
    print("\n✅ ALL MODES LOADED INTO CHROMADB!")