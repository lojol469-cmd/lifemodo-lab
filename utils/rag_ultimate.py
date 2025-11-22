# utils/rag_ultimate.py  ←  Version finale, ultra-légère, 100 % compatible avec ton labo actuel

import os
import json
import numpy as np
import faiss

# Import conditionnel pour rank_bm25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️ rank_bm25 non disponible - BM25 sera désactivé")

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

# ================== CHEMINS ==================
BASE_DIR = "/home/belikan/lifemodo-lab"
CHUNK_DIR = os.path.join(BASE_DIR, "rag_ultimate")
os.makedirs(CHUNK_DIR, exist_ok=True)

# ================== MODÈLES (rien de lourd à télécharger sauf e5 déjà fait)
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device=device)
reranker = None  # Chargé à la demande pour économiser la VRAM

# Variables globales
index = None
bm25 = None
chunks = []
sources = []

def build_or_load_index():
    """À appeler une seule fois au démarrage de Streamlit"""
    global index, bm25, chunks, sources

    index_path = os.path.join(CHUNK_DIR, "faiss.index")
    meta_path = os.path.join(CHUNK_DIR, "meta.json")

    if os.path.exists(index_path):
        print("Chargement du RAG ULTIME existant...")
        index = faiss.read_index(index_path)
        meta = json.load(open(meta_path, encoding="utf-8"))
        chunks = meta["chunks"]
        sources = meta["sources"]
        if BM25_AVAILABLE:
            bm25 = BM25Okapi([c.lower().split() for c in chunks])
        else:
            bm25 = None
            print("⚠️ BM25 désactivé (rank_bm25 non disponible)")
        print(f"RAG chargé → {len(chunks)} chunks prêts")
    else:
        print("Première construction du RAG ULTIME...")
        _build_index_from_dataset()

def _build_index_from_dataset():
    global index, bm25, chunks, sources
    dataset_path = os.path.join(BASE_DIR, "dataset.json")
    if not os.path.exists(dataset_path):
        print("dataset.json pas trouvé → rien à indexer")
        return

    data = json.load(open(dataset_path, encoding="utf-8"))
    index = faiss.IndexFlatIP(1024)
    chunks = []
    sources = []

    for item in data:
        parts = []
        if item.get("text"):         parts.append(item["text"])
        if item.get("ocr"):          parts.append("OCR détecté : " + item["ocr"])
        if item.get("transcript"):   parts.append("Transcription audio : " + item["transcript"])
        if item.get("pdf_title"):    parts.append(f"Source : {item['pdf_title']}")

        text = "\n\n".join(parts)
        if len(text) > 100:
            chunks.append(text)
            src = item.get("image") or item.get("audio_path") or "inconnu"
            sources.append(os.path.basename(src))

    # Embeddings
    print(f"Calcul des embeddings pour {len(chunks)} chunks...")
    embeddings = embedder.encode(chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    index.add(embeddings.astype(np.float32))

    # BM25
    if BM25_AVAILABLE:
        bm25 = BM25Okapi([c.lower().split() for c in chunks])
    else:
        bm25 = None
        print("⚠️ BM25 désactivé (rank_bm25 non disponible)")

    # Sauvegarde
    faiss.write_index(index, os.path.join(CHUNK_DIR, "faiss.index"))
    json.dump({"chunks": chunks, "sources": sources}, open(os.path.join(CHUNK_DIR, "meta.json"), "w", encoding="utf-8"))
    print(f"RAG ULTIME construit et sauvegardé → {len(chunks)} chunks")

# ================== RECHERCHE HYBRIDE ==================
def hybrid_search(query, k=40):
    if index is None or len(chunks) == 0:
        return [], []

    q_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q_emb, k*2)

    # Recherche BM25 si disponible
    bm25_top = []
    if BM25_AVAILABLE and bm25 is not None:
        q_tokens = query.lower().split()
        bm25_scores = np.array(bm25.get_scores(q_tokens))
        bm25_top = np.argsort(bm25_scores)[-k*2:][::-1]

    # Fusion RRF (Reciprocal Rank Fusion)
    score = np.zeros(len(chunks))
    for rank, idx in enumerate(I[0]):   score[idx] += 1 / (rank + 60)
    for rank, idx in enumerate(bm25_top): score[idx] += 1 / (rank + 60)

    top_idx = np.argsort(score)[-k:][::-1]
    return [chunks[i] for i in top_idx], [sources[i] for i in top_idx]

# ================== RE-RANKING + SELF-CORRECTION ==================
def rerank_and_correct(question, candidates):
    global reranker
    if reranker is None:
        reranker = pipeline("text-generation",
                             model="mistralai/Mistral-7B-Instruct-v0.2",
                             torch_dtype=torch.float16,
                             device_map="auto",
                             max_new_tokens=1)

    # Re-ranking top 5
    prompt_rerank = f"""Classe du plus pertinent (1) au moins pertinent pour cette question.
Réponds UNIQUEMENT par les numéros séparés par virgule.

Question : {question}

1. {candidates[0][:600]}
2. {candidates[1][:600]}
3. {candidates[2][:600]}
4. {candidates[3][:600]}
5. {candidates[4][:600]}

Ordre :"""
    try:
        order = reranker(prompt_rerank)[0]["generated_text"]
        nums = [int(x)-1 for x in order.replace("Ordre","").replace(":","").replace(" ","").split(",") if x.isdigit()]
        candidates = [candidates[i] for i in nums if i < len(candidates)]
    except:
        pass  # si ça plante, on garde l'ordre initial

    context = "\n\n".join([f"[Source {i+1}] {c[:1500]}" for i, c in enumerate(candidates[:10])])

    final_prompt = f"""Tu es l’ingénieur gabonais le plus fort du monde en sport automobile et robotique.
Réponds avec une précision extrême en t’appuyant EXCLUSIVEMENT sur les sources ci-dessous.

Sources :
{context}

Question : {question}

Réponse ultra-précise (jargon technique autorisé) :"""

    pipe = pipeline("text-generation",
                     model="mistralai/Mistral-7B-Instruct-v0.2",
                     torch_dtype=torch.float16,
                     device_map="auto",
                     max_new_tokens=1024,
                     do_sample=False)

    answer = pipe(final_prompt)[0]["generated_text"]
    answer = answer.split("Réponse ultra-précise")[ -1].split("Réponse :")[-1].strip()

    return answer, candidates[:8], context

# ================== FONCTION PRINCIPALE À UTILISER PARTOUT ==================
def ask_gabon(question: str):
    """Utilise cette fonction partout dans ton labo"""
    candidates, files = hybrid_search(question, k=40)
    if not candidates:
        return "Aucun contexte trouvé – charge ton dataset.json d’abord.", [], []

    answer, used_chunks, context = rerank_and_correct(question, candidates)
    return answer, used_chunks, files[:8]

# Auto-construction au premier lancement
build_or_load_index()