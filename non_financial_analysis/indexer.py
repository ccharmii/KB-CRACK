# # /KB-CRACK/non_financial_analysis/indexer.py
# 텍스트를 청킹하고 임베딩 -> FAISS 인덱스를 저장 및 재사용

import os
import uuid
from typing import List, Dict

from tqdm import tqdm
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from .config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, INDEX_DIR, TOP_K


DEFAULT_TOKEN_BUDGET = 240_000
MAX_ITEMS_PER_BATCH = 32


def _get_encoder():
    """임베딩 모델 기준 토큰 인코더 반환"""
    try:
        return tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


_enc = _get_encoder()


def _ntokens(text: str) -> int:
    """텍스트의 토큰 길이 추정"""
    return len(_enc.encode(text or ""))


def _batches(texts: List[str], metas: List[Dict]):
    """배치 토큰 및 건수 한도를 지키는 텍스트 배치 생성기"""
    cur_t: List[str] = []
    cur_m: List[Dict] = []
    cur_tok = 0

    for t, m in zip(texts, metas):
        nt = _ntokens(t)

        if nt > DEFAULT_TOKEN_BUDGET:
            if cur_t:
                yield cur_t, cur_m
                cur_t, cur_m, cur_tok = [], [], 0
            yield [t], [m]
            continue

        if cur_t and (cur_tok + nt > DEFAULT_TOKEN_BUDGET or len(cur_t) >= MAX_ITEMS_PER_BATCH):
            yield cur_t, cur_m
            cur_t, cur_m, cur_tok = [], [], 0

        cur_t.append(t)
        cur_m.append(m)
        cur_tok += nt

    if cur_t:
        yield cur_t, cur_m


def build_or_load_faiss(corp_root: str):
    """FAISS 인덱스 디렉터리를 준비하고 기존 인덱스를 로드"""
    idx_dir = os.path.join(corp_root, INDEX_DIR)
    os.makedirs(idx_dir, exist_ok=True)

    idx_path = os.path.join(idx_dir, "faiss_index")
    if os.path.exists(idx_path):
        return FAISS.load_local(
            idx_path,
            OpenAIEmbeddings(model=EMBEDDING_MODEL),
            allow_dangerous_deserialization=True,
        )
    return None


def ingest_texts(corp_root: str, files: List[Dict], corp_code: str):
    """
    파일 목록을 청킹하고 임베딩하여 FAISS 인덱스를 구축 및 저장
    입력: corp_root(루트 경로), files(파일 경로 및 메타데이터 목록), corp_code(법인 코드)
    출력: vs(벡터스토어), texts(청크 텍스트 목록), metadatas(청크 메타데이터 목록)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    texts: List[str] = []
    metadatas: List[Dict] = []

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
        if vs is None:
            vs = FAISS.from_texts(btexts, embeddings, metadatas=bmetas)
        else:
            vs.add_texts(btexts, metadatas=bmetas)

    vs.save_local(os.path.join(corp_root, INDEX_DIR, "faiss_index"))
    return vs, texts, metadatas


def get_retriever(vs, quarter, top_k: int = TOP_K):
    """
    분기 필터링과 MMR을 결합한 검색 함수 반환
    입력: vs(벡터스토어), quarter(분기 문자열), top_k(반환 문서 수)
    출력: query 문자열을 받아 문서 리스트를 반환하는 검색 함수
    """

    def _search(q: str):
        q1 = f"{q} {quarter} 해당 분기"
        docs1 = vs.max_marginal_relevance_search(
            q1,
            k=top_k,
            fetch_k=top_k * 10,
            lambda_mult=0.5,
        )
        hits = [d for d in docs1 if d.metadata.get("quarter") == quarter]

        if len(hits) < top_k:
            docs2 = vs.max_marginal_relevance_search(
                q,
                k=top_k * 2,
                fetch_k=top_k * 10,
                lambda_mult=0.3,
            )
            hits += [d for d in docs2 if d.metadata.get("quarter") == quarter]

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
