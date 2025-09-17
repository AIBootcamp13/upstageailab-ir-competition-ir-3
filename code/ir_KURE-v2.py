import os
import json
from typing import List, Set
from typing_extensions import TypedDict

from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from kiwipiepy import Kiwi

# 1) 리소스 준비
EMBED_MODEL = 'nlpai-lab/KURE-v1'
FAISS_DIR1 = "faiss_index_question"
FAISS_DIR2 = "faiss_index_KURE-v1_semantic"

embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore1 = FAISS.load_local(FAISS_DIR1, embeddings=embed, allow_dangerous_deserialization=True)
retriever1 = vectorstore1.as_retriever(search_kwargs={"k": 10})  # 여유 있게 더 가져오기
vectorstore2 = FAISS.load_local(FAISS_DIR2, embeddings=embed, allow_dangerous_deserialization=True)
retriever2 = vectorstore2.as_retriever(search_kwargs={"k": 10})  # 여유 있게 더 가져오기

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
bm25retriever.k = 3

# 2) 유틸: 문서 -> 텍스트
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# 3) 질의에 관련된 상위 3개 컨텍스트 반환
def top3_contexts(question: str) -> List[str]:
    # 방법 A: retriever 사용
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs: List[Document] = retriever.invoke(question)  # LangChain Runnable 인터페이스
    return [d.metadata["docid"] for d in docs]

    # 방법 B: 벡터스토어 직접 호출(동일 결과)
    # docs_scores = vectorstore.similarity_search_with_score(question, k=3)
    # return [d.page_content for d, _ in docs_scores]

# 4) 중복된 docid 제거
def top3_unique_docids(question: str) -> List[str]:
 # Initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2], weights=[0.1, 0.9]
    )

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


if __name__ == "__main__":
    input_path = "/data/ephemeral/home/dev/IR/data/eval_with_intent.jsonl"
    output_path = "/data/ephemeral/home/dev/IR/data/55_30samp_eval_with_KURE-v1_submissuion_ques.jsonl"
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
                rec['topk'] = top3_unique_docids(q)
                for i, ctx in enumerate(rec['topk'], 1):
                    print(f"[{i}] {ctx}\n")
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")  # JSONL로 한 줄씩 저장[1]


    