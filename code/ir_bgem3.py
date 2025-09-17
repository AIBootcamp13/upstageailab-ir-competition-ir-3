import os
import json
from typing import List, Set
from typing_extensions import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# 1) 리소스 준비
EMBED_MODEL = "BAAI/bge-m3"
FAISS_DIR = "faiss_index_bgem3_semantic"

embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = FAISS.load_local(FAISS_DIR, embeddings=embed, allow_dangerous_deserialization=True)

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # 여유 있게 더 가져오기
    docs: List[Document] = retriever.invoke(question)

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
    output_path = "/data/ephemeral/home/dev/IR/data/eval_with_intent_submissuion.jsonl"
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


    
# docid 별로 만든 벡터 스토어 검색 결과 
"""
=== Top-3 Contexts ===
[1] c63b9e3a-716f-423a-9c9b-0bcaa1b9f35d

[2] 9712bdf6-9419-4953-a8f1-8a4015dee986

[3] 29f939e1-a784-40fc-a31b-139fdaceec66
"""
# 시만틱 청크 벡터 스토어를 이용한 경우 결과
"""
[1] c63b9e3a-716f-423a-9c9b-0bcaa1b9f35d

[2] c63b9e3a-716f-423a-9c9b-0bcaa1b9f35d

[3] 9712bdf6-9419-4953-a8f1-8a4015dee986
"""
# 유니크 docid 함수를 이용한 결과
"""
=== Top-3 Contexts ===
[1] c63b9e3a-716f-423a-9c9b-0bcaa1b9f35d

[2] 9712bdf6-9419-4953-a8f1-8a4015dee986

[3] e227a022-da3b-4810-9882-a2b27c76cc79
"""
# 관련 없는 쿼리 ("니가 대답을 잘해줘서 너무 신나!") 를 넣었을때 결과
"""
=== Top-3 Contexts ===
[1] 57ce3c6f-9665-46df-862a-3c4f02b010b5

[2] 10468cee-aa65-453f-9372-75ddc11a3b77

[3] 30f6ad1f-d4ae-44e9-bcdb-b0390e470377
"""

# 관련 없는 쿼리를 넣었을 때는 빈 리스트를 반환하도록 쿼리 라우터를 만들어야됨