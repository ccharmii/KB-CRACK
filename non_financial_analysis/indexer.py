# 텍스트 청킹 -> 임베딩 -> FAISS 인덱싱(저장/재사용) -> 리트리버
import os, uuid
import hashlib
from typing import List, Dict, Tuple, Iterator
from tqdm import tqdm
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from .config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, INDEX_DIR, TOP_K

DEFAULT_TOKEN_BUDGET = 240_000
MAX_ITEMS_PER_BATCH  = 32

# 인코더
def _get_encoder():
    try:
        return tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

_enc = _get_encoder()

# 토큰 길이 추정
def _ntokens(text): 
    return len(_enc.encode(text or ""))

# 배치 토큰/건수 한도를 지키며 묶어서 yield
def _batches(texts, metas):
    cur_t: List[str] = []; cur_m: List[Dict] = []; cur_tok = 0
    for t, m in zip(texts, metas):
        nt = _ntokens(t)
        if nt > DEFAULT_TOKEN_BUDGET:
            if cur_t: 
                yield cur_t, cur_m
                cur_t, cur_m, cur_tok = [], [], 0
            yield [t], [m]; continue
        if cur_t and (cur_tok + nt > DEFAULT_TOKEN_BUDGET or len(cur_t) >= MAX_ITEMS_PER_BATCH):
            yield cur_t, cur_m
            cur_t, cur_m, cur_tok = [], [], 0
        cur_t.append(t); cur_m.append(m); cur_tok += nt
    if cur_t: yield cur_t, cur_m

# 생성, 이미 존재 시 로드
def build_or_load_faiss(corp_root: str):
    idx_dir = os.path.join(corp_root, INDEX_DIR)
    os.makedirs(idx_dir, exist_ok=True)
    idx_path = os.path.join(idx_dir, "faiss_index")
    if os.path.exists(idx_path):
        return FAISS.load_local(idx_path, OpenAIEmbeddings(model=EMBEDDING_MODEL), allow_dangerous_deserialization=True)
    return None

# 청킹 -> 임베딩 -> FAISS 구축
def ingest_texts(corp_root: str, files: List[Dict], corp_code: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts: List[str] = []; metadatas: List[Dict] = []
    for it in tqdm(files, desc="chunking"):
        with open(it["path"], "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        chunks = splitter.split_text(txt)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            m = dict(it["meta"])
            m["chunk_id"] = str(uuid.uuid4())
            m["chunk_idx"] = i
            metadatas.append(m)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, max_retries=6)
    vs = None
    for btexts, bmetas in tqdm(_batches(texts, metadatas), desc="embedding+indexing"):
        if vs is None: vs = FAISS.from_texts(btexts, embeddings, metadatas=bmetas)
        else: vs.add_texts(btexts, metadatas=bmetas)

    vs.save_local(os.path.join(corp_root, INDEX_DIR, "faiss_index"))
    return vs, texts, metadatas

# MMR 기반 리트리버
def get_retriever(vs, quarter, top_k=TOP_K):
    def _search(q: str):

        # 1차: 분기 토큰 강화 + MMR로 다양성 확보
        q1 = f"{q} {quarter} 해당 분기"
        docs1 = vs.max_marginal_relevance_search(
            q1, k=top_k, fetch_k=top_k*10, lambda_mult=0.5
        )
        hits = [d for d in docs1 if d.metadata.get("quarter") == quarter]

        # 부족하면 2차 보충 (MMR 파라미터 변화로 다른 각도 탐색)
        if len(hits) < top_k:
            docs2 = vs.max_marginal_relevance_search(
                q, k=top_k*2, fetch_k=top_k*10, lambda_mult=0.3
            )
            hits += [d for d in docs2 if d.metadata.get("quarter") == quarter]

        # 중복 제거(동일 chunk_id 제거) 후 상위 top_k 반환
        seen = set()
        uniq = []
        for d in hits:
            key = (d.metadata.get("rcept_no"), d.metadata.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(d)
            if len(uniq) >= top_k:
                break
        return uniq
    return _search
