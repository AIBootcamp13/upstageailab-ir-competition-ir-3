import os
import json
from typing import List, Set, Dict
from typing_extensions import TypedDict

from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from kiwipiepy import Kiwi

# 1) 리소스 준비
EMBED_MODEL = 'nlpai-lab/KURE-v1'
FAISS_DIR = "faiss_index_KURE-v1_semantic"

embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = FAISS.load_local(FAISS_DIR, embeddings=embed, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # 여유 있게 더 가져오기

# BM25 retrieve
kiwi_tokenizer = Kiwi()

def kiwi_tokenize(text):
    return [token.form for token in kiwi_tokenizer.tokenize(text)]

def meta_mapper(record: dict, metadata: dict) -> dict:
        # record: jq_schema로 추출된 해당 JSON 객체
        # metadata: 기본 메타데이터(source, seq_num 등)
        metadata.update({
            "docid": record.get("docid"),
            "src": record.get("src"),
        })
        return metadata

loader = JSONLoader(
    file_path="/data/ephemeral/home/dev/IR/data/documents.jsonl",
    jq_schema=".",               # 한 줄 = 한 객체 전체
    content_key="content",       # 그 객체에서 content를 page_content로 사용
    is_content_key_jq_parsable=False,
    metadata_func=meta_mapper,   # docid, src를 metadata로 보존
    text_content=False,          # content가 텍스트이므로 False로 두면 전체 객체를 보존해도 됨
    json_lines=True,             # JSON Lines 파일
)

docs = loader.load()
    
bm25retriever = BM25Retriever.from_documents(docs, preprocess_function=kiwi_tokenize)
bm25retriever.k = 10

# reranker 정의
reranker = HuggingFaceCrossEncoder(
    model_name='BAAI/bge-reranker-v2-m3',
)


# 2) 유틸: 문서 -> 텍스트
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# 3) 질의에 관련된 상위 3개 컨텍스트 반환
def top3_contexts(question: str) -> List[str]:
    # 방법 A: retriever 사용
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # retriever 함수는 k값에 따라 랭체인 표준 객체 document의 리스트를 반환한다.
    docs: List[Document] = retriever.invoke(question)  # LangChain Runnable 인터페이스
    return [d.metadata["docid"] for d in docs]

    # 방법 B: 벡터스토어 직접 호출(동일 결과)
    # docs_scores = vectorstore.similarity_search_with_score(question, k=3)
    # return [d.page_content for d, _ in docs_scores]

# 4) 중복된 docid 제거
def top3_unique_docids(question: str) -> List[str]:
 # Initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25retriever, retriever], weights=[0.2, 0.8]
    )
    docs: List[Document] = retriever.invoke(question)
    docs: List[Document] = bm25retriever.invoke(question)
    docs: List[Document] = ensemble_retriever.invoke(question)

    seen: Set[str] = set()
    result: List[str] = []
    for d in docs:
        docid = d.metadata.get("docid")
        if docid is None:
            continue
        if docid in seen:
            continue
        seen.add(docid)
        result.append(docid)
        if len(result) == 3:
            break
    return result

# 5. reranker
"""
"""
def rerank(question: str) -> List[str]:
    # 1) 두 리트리버에서 후보 수집
    base_docs: List[Document] = []
    try:
        base_docs.extend(retriever.invoke(question) or [])
    except Exception:
        pass
    try:
        base_docs.extend(bm25retriever.invoke(question) or [])
    except Exception:
        pass

    if not base_docs:
        return []

     # 2) docid 기준으로 중복 제거하며 최고 점수를 유지
    best_by_docid: Dict[str, Tuple[float, Document]] = {}
    for doc in base_docs:
        docid = (doc.metadata or {}).get("docid")
        if not docid:
            # 식별자가 없으면 스킵(정책에 따라 허용할 수도 있음)
            continue
        try:
            lst = [doc.page_content or "", question]
            score = float(reranker.score(lst))
        except Exception:
            # 점수 계산 실패 시 매우 낮은 점수 부여 또는 스킵
            score = float("-inf")

        # 동일 docid면 더 높은 점수를 유지
        prev = best_by_docid.get(docid)
        if (prev is None) or (score > prev[0]):
            best_by_docid[docid] = (score, doc)

    if not best_by_docid:
        return []

        # 3) 점수로 내림차순 정렬 후 상위 3개 docid 반환
    ranked = sorted(best_by_docid.items(), key=lambda kv: kv[1][0], reverse=True)
    top_docids = [docid for docid, (_score, _doc) in ranked[:3]]
    return top_docids



if __name__ == "__main__":
    input_path = "/data/ephemeral/home/dev/IR/data/eval_with_intent_mod.jsonl"
    output_path = "/data/ephemeral/home/dev/IR/data/eval_with_rerank_submissuion_5_.jsonl"
    with (
        open(input_path, "r", encoding="utf-8") as f_in, 
        open(output_path, "w", encoding="utf-8") as f_out
    ):
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)  # 각 줄을 dict로 파싱[1]
            id = rec.get("docid")
            greet_id = [276, 261, 283, 32, 94, 90, 220, 245, 229, 247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218] 
            if id in greet_id:
                rec['topk'] = []
            else:
                q = rec.get("standalone_question")
                rec['topk'] = rerank(q)
                for i, ctx in enumerate(rec['topk'], 1):
                    print(f"[{i}] {ctx}\n")
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")  # JSONL로 한 줄씩 저장[1]

