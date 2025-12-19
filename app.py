import os
from dotenv import load_dotenv
import streamlit as st
from pinecone import Pinecone,ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter



# Setup & initialization:

load_dotenv()

# Setting the Pinecone API Key:
PINECONE_API_KEY=os.getenv("pinecone_api_key")
INDEX_NAME="hybrid-search-langchain-pinecone"


@st.cache_resource
def init_pinecone_and_embeddings():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables.")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(INDEX_NAME)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return pc, index, embeddings


def load_docs_from_uploads(uploaded_files):
    os.makedirs("temp_uploads", exist_ok=True)
    docs = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        ext = os.path.splitext(file_name)[1].lower()
        temp_path = os.path.join("temp_uploads", file_name)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        if ext == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif ext in [".txt", ".md"]:
            loader = TextLoader(temp_path, encoding="utf-8")
        elif ext == ".csv":
            loader = CSVLoader(temp_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(temp_path)
        else:
            continue

        docs.extend(loader.load())

    return docs


def build_retriever(index, embeddings, docs):
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    bm25_encoder = BM25Encoder().default()
    bm25_encoder.fit(texts)

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=index,
    )

    retriever.add_texts(texts, metadatas=metadatas)

    return retriever



# Streamlit Application
st.set_page_config(page_title="Hybrid Search RAG Demo", page_icon="🧠",layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #111827 0, #020617 35%, #020617 100%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    body, .block-container {
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .app-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #f9fafb;
        letter-spacing: 0.03em;
        margin-bottom: 0.25rem;
    }

    .app-subtitle {
        font-size: 1.05rem;
        color: #cbd5f5;
        margin-bottom: 1.1rem;
    }

    .section-card {
        background: rgba(15,23,42,0.96);
        border-radius: 18px;
        padding: 1.6rem 1.7rem;
        border: 1px solid rgba(148,163,184,0.55);
        box-shadow: 0 18px 45px rgba(15,23,42,0.75);
        margin-bottom: 1.6rem;
        backdrop-filter: blur(18px);
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #f9fafb;
    }

    .section-subtitle {
        font-size: 0.95rem;
        color: #9ca3e6;
        margin-bottom: 0.9rem;
    }

    .stButton>button {
        border-radius: 999px;
        border: 1px solid rgba(250,204,21,0.75);
        padding: 0.55rem 1.9rem;
        background: linear-gradient(110deg, #f97316, #eab308);
        color: #020617;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 12px 30px rgba(250,204,21,0.4);
    }

    .stButton>button:hover {
        background: linear-gradient(110deg, #fdba74, #facc15);
        border-color: #facc15;
    }

    .stRadio>div {
        background: rgba(15,23,42,1);
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.75);
    }

    .stRadio label {
        padding-right: 0.7rem;
        font-size: 0.9rem;
        color: #e5e7eb;
    }

    .stSlider>div[data-baseweb="slider"]>div>div {
        background: rgba(55,65,81,0.7);
    }
    .stSlider>div[data-baseweb="slider"]>div>div>div {
        background: #facc15;
    }

    .stTextInput>div>div>input {
        background-color: #020617;
        color: #e5e7eb;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.8);
        padding: 0.45rem 0.9rem;
    }

    .stTextInput>div>div>input::placeholder {
        color: #6b7280;
    }

    .stFileUploader label {
        color: #e5e7eb;
    }

    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background-color: #020617;
        border: 1px dashed rgba(148,163,184,0.8);
    }

    .stSubheader, h2, h3 {
        color: #f9fafb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="app-title">🧠 Hybrid Search RAG with Pinecone & LangChain</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload your documents and query them with a hybrid of dense embeddings and BM25 keyword search.</div>',
    unsafe_allow_html=True,
)

with st.expander("How it works"):
    st.markdown(
        """
1. You upload PDFs, Word, CSV or text files.
2. The app splits them into overlapping chunks and stores them in Pinecone.
3. For each chunk it builds:
   - a dense embedding with MiniLM,
   - a sparse BM25 vector based on terms.
4. At query time it combines both signals (hybrid search), so you get:
   - keyword precision from BM25,
   - semantic understanding from embeddings.
        """
    )

pc, index, embeddings = init_pinecone_and_embeddings()

with st.container():
    st.markdown(
        '<div class="section-subtitle">Drag and drop PDFs, Word files, CSVs, text, or Markdown. They will be chunked and indexed in Pinecone.</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "",
        type=["pdf", "txt", "md", "csv", "docx"],
        accept_multiple_files=True,
    )

    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None

    if st.button("Process documents"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Reading and indexing documents..."):
                docs = load_docs_from_uploads(uploaded_files)
                retriever = build_retriever(index, embeddings, docs)
                st.session_state["retriever"] = retriever

            if st.session_state["retriever"] is None:
                st.error("Could not build retriever. Check file types and try again.")
            else:
                st.success("Documents processed and indexed successfully.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-subtitle">Choose the retrieval mode, adjust the hybrid weight, and run your query over the indexed documents.</div>',
    unsafe_allow_html=True,
)

mode = st.radio(
    "Retrieval mode",
    ["Hybrid", "Dense only", "Sparse only"],
    horizontal=True,
)

alpha = st.slider(
    "Hybrid weight (0 = more sparse, 1 = more dense)",
    0.0,
    1.0,
    0.5,
    0.05,
)

query = st.text_input("Your question", placeholder="e.g. Summarize the contract terms")
top_k = st.slider("Number of results to retrieve", 1, 10, 3)

if st.button("Search") and query.strip():
    retriever = st.session_state.get("retriever", None)

    if retriever is None:
        st.warning("Please upload and process documents first.")
    else:
        if mode == "Hybrid":
            retriever.alpha = alpha
        elif mode == "Dense only":
            retriever.alpha = 1.0
        else:
            retriever.alpha = 0.0

        with st.spinner("Searching with hybrid retrieval..."):
            docs = retriever.invoke(query)
    

        st.subheader("Retrieved chunks")
        if not docs:
            st.write("No results found.")
        else:
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Result {i}:**")
                st.write(doc.page_content)
                if doc.metadata:
                    st.caption(str(doc.metadata))
                st.markdown("---")

st.markdown("</div>", unsafe_allow_html=True)