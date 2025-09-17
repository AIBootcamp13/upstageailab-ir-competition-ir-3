import os
import json
import heapq
import math

from typing import List, Dict, Any, Iterable, Tuple
from langchain.schema import Document
from langchain_community.vectorstores import FAISS


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
# retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # 여유 있게 더 가져오기

# reranker 정의
reranker = HuggingFaceCrossEncoder(
    model_name='BAAI/bge-reranker-v2-m3',
)


def iter_all_docs(vs: FAISS = vectorstore):
    for dsid in vs.index_to_docstore_id.values():  # docstore_id 문자열들[1]
        doc = vs.docstore.search(dsid)  # 구현체에 따라 get/search 메서드 제공[1]
        if doc is not None:
            yield doc

def safe_score(reranker, text: str, query: str) -> float:
    try:
        lst = [text or "", query]   
        return float(reranker.score(lst))
    except Exception:
        # 점수 계산 실패 시 매우 낮은 점수 부여 또는 스킵
        return float("-inf")
    
def rerank_over_all(question: str, top_k: int = 3) -> List[str]:
    # 최소 힙에 (score, docid) 저장. 상위 점수만 유지하기 위해 음수 변환 또는 튜플 활용
    # 여기서는 (score, docid, doc)로 유지하고, 힙 크기를 top_k로 제한
    heap: List[Tuple[float, str, Document]] = []

    for doc in iter_all_docs():
        # docid 확보
        docid = (doc.metadata or {}).get("docid")
        if not docid:
            continue

        score = safe_score(reranker, doc.page_content or "", question)
        if math.isinf(score) and score < 0:
            continue

        # 힙에 top_k 개만 유지: 가장 낮은 점수는 제거
        if len(heap) < top_k:
            heapq.heappush(heap, (score, docid, doc))
        else:
            # 최솟값보다 크면 교체
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, docid, doc))

    if not heap:
        return []

    # 힙은 최소 힙이므로 내림차순으로 정렬
    heap.sort(key=lambda x: x[0], reverse=True)
    print(heap)
    return [docid for (_score, docid, _doc) in heap]



if __name__ == "__main__":
    input_path = "/data/ephemeral/home/dev/IR/data/eval_with_intent_mod.jsonl"
    output_path = "/data/ephemeral/home/dev/IR/data/eval_with_rerank_submissuion_3_all.jsonl"
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
                rec['topk'] = rerank_over_all(q)
                for i, ctx in enumerate(rec['topk'], 1):
                    print(f"[{i}] {ctx}\n")
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")  # JSONL로 한 줄씩 저장[1]

