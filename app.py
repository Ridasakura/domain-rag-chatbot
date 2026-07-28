"""
Stacks AI — Enterprise Hybrid RAG Suite (v4.0)
Featuring Unified ChromaDB Retrieval, BM25 + Cross-Encoder Reranking, SHA-256 Deduplication,
Grounded Context Injection, Query Expansion, and Robust Error Handling.
"""

import os
import io
import json
import time
import re
import hashlib
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from groq import Groq

# Optional OCR Support
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ---------------------------------------------------------------
# Page Configuration & Modern Styling
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Stacks AI — Enterprise Hybrid RAG Suite",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

:root {
    --primary-indigo: #6366F1;
    --radius-lg: 14px;
    --radius-sm: 8px;
}

.stacks-header {
    padding: 1rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.2);
}
.stacks-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: var(--primary-indigo);
    text-transform: uppercase;
    font-weight: 600;
}
.stacks-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0.2rem 0;
}

.status-card {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: var(--radius-lg);
    padding: 0.8rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background-color: rgba(128, 128, 128, 0.03);
}
.status-indicator {
    width: 10px; height: 10px; border-radius: 50%; background-color: #9CA3AF;
}
.status-indicator.active {
    background-color: #10B981; box-shadow: 0 0 8px #10B98188;
}

.chat-container {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: var(--radius-lg);
    padding: 1.2rem;
    background-color: rgba(128, 128, 128, 0.02);
    margin-bottom: 0.8rem;
}

.citation-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 10px 14px;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(99, 102, 241, 0.25);
    background-color: rgba(99, 102, 241, 0.04);
    margin: 6px 0;
}

.flashcard-box {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: var(--radius-lg);
    padding: 1.2rem;
    min-height: 110px;
    background-color: rgba(128, 128, 128, 0.03);
    margin-bottom: 0.5rem;
}
</style>

<div class="stacks-header">
    <div class="stacks-eyebrow">Enterprise Hybrid RAG Suite · Unified ChromaDB Core</div>
    <div class="stacks-title">🎓 Stacks AI</div>
</div>
""", unsafe_allow_html=True)

SYSTEM_PROMPT = """You are a highly precise document assistant.
Answer the question using ONLY the provided context excerpts below.
For every key claim or fact you state, cite its exact source tag in square brackets, e.g., [Doc: Physics.pdf, Page 12].
If the answer cannot be determined strictly from the provided context, state:
"I could not find this information in the uploaded documents."
Do not extrapolate or use external facts."""


# ---------------------------------------------------------------
# Core Cached Model Loaders
# ---------------------------------------------------------------
@st.cache_resource
def load_dense_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./stacks_chroma_db")


# ---------------------------------------------------------------
# Document Parsing, Deduplication & Indexing
# ---------------------------------------------------------------
def compute_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def extract_pdf_pages(file, file_bytes, enable_ocr=False):
    pages = []
    reader = PdfReader(io.BytesIO(file_bytes))
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        
        # Fallback OCR for empty/scanned pages
        if not text.strip() and enable_ocr and HAS_OCR:
            try:
                images = convert_from_bytes(file_bytes, first_page=i+1, last_page=i+1)
                if images:
                    text = pytesseract.image_to_string(images[0])
            except Exception:
                text = ""
                
        if text.strip():
            pages.append({"text": text, "source": file.name, "page": i + 1})
            
    return pages


def safe_json_parse(raw_text):
    try:
        return json.loads(raw_text)
    except Exception:
        pass
    match = re.search(r"(\[.*\]|\{.*\})", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return []


def rewrite_query(user_query, chat_history, client):
    if not chat_history:
        return user_query
    
    recent_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat_history[-4:]])
    prompt = f"""Given the conversation history and follow-up question, rewrite it as a standalone search query. Do NOT answer the question.

History:
{recent_history}

Follow-up Question: {user_query}
Standalone Search Query:"""

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return user_query


def build_unified_index(uploaded_files, embedder, enable_ocr, progress_bar, status_text):
    start_time = time.time()
    chroma_client = get_chroma_client()
    
    # Recreate Collection on Reset
    try:
        chroma_client.delete_collection("stacks_collection")
    except Exception:
        pass
    collection = chroma_client.create_collection("stacks_collection")

    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=120)
    all_chunks = []
    processed_hashes = set()
    total_pages = 0

    status_text.text("📖 Processing PDFs & checking duplicates...")
    progress_bar.progress(20)

    for f in uploaded_files:
        f_bytes = f.read()
        f_hash = compute_file_hash(f_bytes)
        f.seek(0)

        if f_hash in processed_hashes:
            continue
        processed_hashes.add(f_hash)

        pages = extract_pdf_pages(f, f_bytes, enable_ocr=enable_ocr)
        total_pages += len(pages)
        for page_data in pages:
            for t in splitter.split_text(page_data["text"]):
                all_chunks.append({
                    "text": t,
                    "source": page_data["source"],
                    "page": page_data["page"]
                })

    if not all_chunks:
        return None, [], 0, 0, 0

    status_text.text("⚡ Building BM25 Index & Vectorizing into ChromaDB...")
    progress_bar.progress(60)

    # BM25 Keyword Engine
    tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)

    # Dense Vector Persistence in ChromaDB
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"], "page": c["page"]} for c in all_chunks],
        ids=[f"chunk_{i}" for i in range(len(all_chunks))]
    )

    elapsed_time = round(time.time() - start_time, 2)
    progress_bar.progress(100)
    status_text.text("✅ Unified Index Active!")
    
    return bm25_index, all_chunks, total_pages, len(embeddings), elapsed_time


def hybrid_retrieve_and_rerank(query, embedder, reranker, bm25_index, chunks, target_doc="All Documents", final_k=4):
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection("stacks_collection")
    
    # 1. Vector Search via ChromaDB
    q_emb = embedder.encode([query], convert_to_numpy=True).tolist()
    where_filter = {"source": target_doc} if target_doc != "All Documents" else None

    chroma_results = collection.query(
        query_embeddings=q_emb,
        n_results=min(15, len(chunks)),
        where=where_filter
    )
    
    candidate_indices = set()
    if chroma_results and chroma_results["ids"]:
        for cid in chroma_results["ids"][0]:
            candidate_indices.add(int(cid.split("_")[1]))

    # 2. Keyword Search via BM25
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:15]

    for idx in top_bm25_indices:
        if target_doc == "All Documents" or chunks[idx]["source"] == target_doc:
            candidate_indices.add(idx)

    candidates = [chunks[i] for i in candidate_indices]
    if not candidates:
        return []

    # 3. Cross-Encoder Reranking
    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)

    reranked = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, rerank_scores)]
    return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)[:final_k]


# ---------------------------------------------------------------
# Session State Engine
# ---------------------------------------------------------------
def init_session():
    defaults = {
        "messages": [], "bm25_index": None, "chunks": [],
        "uploaded_files": [], "flashcards": [], "key_concepts": "",
        "quiz_data": [], "stats": {"pages": 0, "embeddings": 0, "time": 0.0}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()
default_key = st.secrets.get("GROQ_API_KEY", "")

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Stacks AI Studio")
    st.caption("Hybrid RAG Platform (ChromaDB Core)")
    st.divider()

    st.markdown("**1. Parsing Options**")
    enable_ocr = st.checkbox("Enable OCR Fallback", value=False)
    
    st.markdown("**2. Document Upload**")
    uploaded_files = st.file_uploader("Upload Course PDFs", type=["pdf"], accept_multiple_files=True)
    process_clicked = st.button("⚡ Build Index", use_container_width=True)

    st.divider()
    st.markdown("**📊 Index Stats**")
    st.text(f"• Documents: {len(st.session_state.uploaded_files)}")
    st.text(f"• Pages Indexed: {st.session_state.stats['pages']}")
    st.text(f"• Total Chunks: {len(st.session_state.chunks)}")
    st.text(f"• Build Time: {st.session_state.stats['time']}s")

    st.divider()
    if st.button("🗑 Reset Database & Session", use_container_width=True):
        try:
            get_chroma_client().delete_collection("stacks_collection")
        except Exception:
            pass
        for k in ["messages", "bm25_index", "chunks", "uploaded_files", "flashcards", "key_concepts", "quiz_data"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

embedder = load_dense_embedder()
reranker = load_cross_encoder()

if process_clicked and uploaded_files:
    p_bar = st.sidebar.progress(0)
    s_txt = st.sidebar.empty()
    bm25_idx, chunks, pages, embeddings, proc_time = build_unified_index(
        uploaded_files, embedder, enable_ocr, p_bar, s_txt
    )
    st.session_state.bm25_index = bm25_idx
    st.session_state.chunks = chunks
    st.session_state.uploaded_files = uploaded_files
    st.session_state.stats = {"pages": pages, "embeddings": embeddings, "time": proc_time}
    p_bar.empty()
    s_txt.empty()
    st.rerun()

# ---------------------------------------------------------------
# Dashboard Readiness Indicator
# ---------------------------------------------------------------
is_ready = st.session_state.bm25_index is not None
cols = st.columns(4)
for i, (label, active) in enumerate([
    ("Documents", is_ready), ("ChromaDB Engine", is_ready),
    ("Flashcards", len(st.session_state.flashcards) > 0),
    ("Quiz Engine", len(st.session_state.quiz_data) > 0)
]):
    with cols[i]:
        st.markdown(f'''
        <div class="status-card">
            <div class="status-indicator {'active' if active else ''}"></div>
            <div><div class="status-label">{label}</div><span style="font-size:0.8rem;">{"Active" if active else "Idle"}</span></div>
        </div>''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Application Tabs
# ---------------------------------------------------------------
tab_chat, tab_cards, tab_concepts, tab_quiz, tab_settings = st.tabs(
    ["💬 Hybrid Chat", "🎴 Flashcards & Anki", "📌 Key Concepts", "✍️ Practice Quiz", "⚙️ Settings"]
)

# --- TAB: Settings ---
with tab_settings:
    st.subheader("API Access Configuration")
    groq_key_input = st.text_input("Groq API Key", value=default_key, type="password")
    active_groq_key = groq_key_input if groq_key_input else default_key

# --- TAB 1: Chat Q&A ---
with tab_chat:
    doc_scope = st.selectbox("📌 Metadata Filter: Target Document Scope", options=["All Documents"] + [f.name for f in st.session_state.uploaded_files]) if is_ready else "All Documents"

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f'<div class="chat-container">{msg["content"]}</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask a question about your uploaded materials...")

    if question:
        if not is_ready:
            st.error("Please upload PDFs and build the index first.")
        elif not active_groq_key:
            st.error("Please provide a Groq API Key under Settings.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    client = Groq(api_key=active_groq_key)
                    expanded_query = rewrite_query(question, st.session_state.messages[:-1], client)
                    
                    retrieved = hybrid_retrieve_and_rerank(
                        expanded_query, embedder, reranker,
                        st.session_state.bm25_index, st.session_state.chunks,
                        target_doc=doc_scope
                    )
                    
                    if not retrieved:
                        st.warning("No relevant information found.")
                    else:
                        # Construct ground truth context block
                        context_blocks = []
                        for c in retrieved:
                            context_blocks.append(f"[Doc: {c['source']}, Page {c['page']}]\n{c['text']}")
                        
                        full_context = "\n\n".join(context_blocks)[:3500] # Truncate token budget safety
                        prompt = f"Context Excerpts:\n{full_context}\n\nQuestion: {question}"

                        stream = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                            temperature=0.1, stream=True
                        )
                        
                        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                        response_text = st.write_stream(stream)
                        st.markdown('</div>', unsafe_allow_html=True)

                        with st.expander("📄 View Grounded Search Analytics & Sources"):
                            st.caption(f"**Expanded Search Query:** `{expanded_query}`")
                            for item in retrieved:
                                st.markdown(f'<div class="citation-box">📄 <b>{item["source"]}</b> | Page {item["page"]}<br><i>"{item["text"][:140]}..."</i><br><b>Rerank Score:</b> {item["rerank_score"]:.4f}</div>', unsafe_allow_html=True)

                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"Error executing LLM query pipeline: {str(e)}")

# --- TAB 2: Flashcards & Anki Export ---
with tab_cards:
    st.subheader("Retrieval-Based Flashcard Deck")
    if is_ready and active_groq_key:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            card_doc = st.selectbox("Document Scope", options=["All Documents"] + [f.name for f in st.session_state.uploaded_files], key="fc_doc")
        with c2:
            card_count = st.slider("Number of Cards", 5, 25, 10, step=5)
        with c3:
            if st.button("✨ Generate Cards", use_container_width=True):
                try:
                    client = Groq(api_key=active_groq_key)
                    top_chunks = hybrid_retrieve_and_rerank("important definitions formulas concepts overview", embedder, reranker, st.session_state.bm25_index, st.session_state.chunks, target_doc=card_doc, final_k=8)
                    sample_text = "\n\n".join([c["text"] for c in top_chunks])[:3500]
                    
                    prompt = f"Generate {card_count} flashcards as a JSON array with 'question' and 'answer' keys from this context:\n{sample_text}"
                    res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
                    st.session_state.flashcards = safe_json_parse(res.choices[0].message.content)
                except Exception as e:
                    st.error(f"Flashcard generation failed: {str(e)}")

        if st.session_state.flashcards:
            anki_str = "\n".join([f"{fc.get('question','')}\t{fc.get('answer','')}" for fc in st.session_state.flashcards])
            st.download_button("🎴 Export to Anki (.txt)", data=anki_str, file_name="anki_cards.txt", mime="text/plain")
            
            f_cols = st.columns(2)
            for idx, card in enumerate(st.session_state.flashcards):
                with f_cols[idx % 2]:
                    st.markdown(f'<div class="flashcard-box"><b>📘 Q{idx+1}:</b> {card.get("question","")}</div>', unsafe_allow_html=True)
                    with st.expander("Reveal Answer"):
                        st.write(card.get("answer", ""))

# --- TAB 3: Key Concepts ---
with tab_concepts:
    st.subheader("Core Document Concepts")
    if is_ready and active_groq_key:
        concept_doc = st.selectbox("Document Scope", options=["All Documents"] + [f.name for f in st.session_state.uploaded_files], key="kc_doc")
        if st.button("📌 Extract Core Concepts", use_container_width=True):
            try:
                client = Groq(api_key=active_groq_key)
                top_chunks = hybrid_retrieve_and_rerank("key concepts principles summary definitions", embedder, reranker, st.session_state.bm25_index, st.session_state.chunks, target_doc=concept_doc, final_k=10)
                prompt = "Extract key terms, definitions, and formulas as bullet points from context:\n\n" + "\n\n".join([c["text"] for c in top_chunks])[:3500]
                res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
                st.session_state.key_concepts = res.choices[0].message.content
            except Exception as e:
                st.error(f"Concept extraction failed: {str(e)}")

        if st.session_state.key_concepts:
            st.markdown(st.session_state.key_concepts)

# --- TAB 4: Practice Quiz ---
with tab_quiz:
    st.subheader("Adaptive Practice Quiz Engine")
    if is_ready and active_groq_key:
        q1, q2, q3 = st.columns([2, 2, 1])
        with q1:
            quiz_doc = st.selectbox("Document Scope", options=["All Documents"] + [f.name for f in st.session_state.uploaded_files], key="qz_doc")
        with q2:
            quiz_diff = st.selectbox("Difficulty", ["Balanced Mix", "Foundational (Easy)", "Exam Standard (Hard)"])
        with q3:
            if st.button("📝 Synthesize Quiz", use_container_width=True):
                try:
                    client = Groq(api_key=active_groq_key)
                    top_chunks = hybrid_retrieve_and_rerank("core topics test questions problems theory", embedder, reranker, st.session_state.bm25_index, st.session_state.chunks, target_doc=quiz_doc, final_k=8)
                    prompt = f"""Generate a 4-question multiple choice quiz at '{quiz_diff}' level.
                    Return ONLY a JSON array with 'question', 'options' (4 strings), 'correct_index' (0-3), and 'explanation'.
                    Context:\n""" + "\n\n".join([c["text"] for c in top_chunks])[:3500]
                    
                    res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
                    st.session_state.quiz_data = safe_json_parse(res.choices[0].message.content)
                except Exception as e:
                    st.error(f"Quiz generation failed: {str(e)}")

        if st.session_state.quiz_data:
            with st.form("quiz_form"):
                user_answers = []
                for idx, q in enumerate(st.session_state.quiz_data):
                    st.markdown(f"**Q{idx+1}: {q['question']}**")
                    choice = st.radio("Options:", options=q['options'], key=f"q_{idx}")
                    user_answers.append(choice)
                    st.divider()
                
                if st.form_submit_button("Submit Quiz"):
                    score = sum(1 for idx, q in enumerate(st.session_state.quiz_data) if q['options'].index(user_answers[idx]) == q['correct_index'])
                    st.metric("Final Score", f"{score} / {len(st.session_state.quiz_data)}")
