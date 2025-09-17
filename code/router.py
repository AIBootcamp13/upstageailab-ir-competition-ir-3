import os

from typing import List, Literal, TypedDict, Optional
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_upstage import ChatUpstage
from langchain_core.language_models import BaseChatModel 
from langchain.schema import Document

# 1) 의도 분류 프롬프트 (검색 필요 여부만 판단)
ROUTER_PROMPT = """
당신은 사용자의 질문에 대해 '검색 필요 여부'를 판단하는 분류기입니다.

- retrieve: 외부 컨텍스트(벡터 스토어 문서)가 있어야 정확히 답할 수 있는 질문
- no_retrieve: 모델의 상식/상대적으로 일반 지식만으로 충분하거나, 단순 인사/메타 요청

질문: "{question}"

출력은 JSON만(다른 텍스트 금지):
{{
  "intent": "retrieve|no_retrieve",
  "confidence": 0.0~1.0,
  "reason": "간단한 한 문장 근거"
}}
"""

class RouteResult(TypedDict):
    intent: Literal["retrieve", "no_retrieve"]
    confidence: float
    reason: str


def build_router(llm: BaseChatModel):
    schema = [
        ResponseSchema(name="intent", description="retrieve 또는 no_retrieve"),
        ResponseSchema(name="confidence", description="0~1 사이 확신도(float)"),
        ResponseSchema(name="reason", description="판단 근거 한 문장"),
    ]
    parser = StructuredOutputParser.from_response_schemas(schema)
    prompt = PromptTemplate.from_template(ROUTER_PROMPT)
    chain = prompt | llm | parser
    return chain

def decide_retrieval(router_chain, question: str, threshold: float = 0.9) -> RouteResult:
    result = router_chain.invoke({"question": question})
    # 방어적 파싱
    intent = result.get("intent", "retrieve")
    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    reason = result.get("reason", "")
    if intent not in ("retrieve", "no_retrieve"):
        intent = "retrieve"
    # 임계값 아래면 보수적으로 retrieve
    if confidence < threshold:
        intent = "retrieve"
    return {"intent": intent, "confidence": confidence, "reason": reason}

# 2) 검색 실행 함수 (필요 시만 호출)
def maybe_retrieve(question: str, retriever, router_chain, k: int = 3) -> dict:
    route = decide_retrieval(router_chain, question)
    if route["intent"] == "retrieve":
        docs: List[Document] = retriever.invoke(question)
        return {
            "need_retrieval": True,
            "reason": route["reason"],
            "confidence": route["confidence"],
            "docids": [d.metadata.get("docid") for d in docs[:k]],
            "contexts": [d.page_content for d in docs[:k]],
        }
    else:
        return {
            "need_retrieval": False,
            "reason": route["reason"],
            "confidence": route["confidence"],
            "docids": [],
            "contexts": [],
        }

# 3) 사용 예시
if __name__ == "__main__":
    
    os.environ["UPSTAGE_API_KEY"] = ""

    # LLM 초기화
    llm = ChatUpstage(
    model_name="solar-pro2", 
    temperature=0,
    )
    router_chain = build_router(llm)

    # retriever는 기존 vectorstore.as_retriever(search_kwargs={"k": 5}) 등으로 구성되어 있다고 가정
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    embed = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vs = FAISS.load_local("faiss_index_bgem3_semantic", embeddings=embed, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    # 테스트
    queries = [
        "상대성이론이 뭐야?",                # 일반 지식일 수 있으나 애매 → 임계값 낮으면 retrieve
        "이 리포트의 결론 부분만 요약해줘",   # 외부 문서 필요 → retrieve
        "안녕!",                           # 인사 → no_re
    ]

    for q in queries:
        res = maybe_retrieve(q, retriever, router_chain, k=3)
        print(f"Q: {q}")
        print(f"- need_retrieval: {res['need_retrieval']} (conf={res['confidence']:.2f})")
        print(f"- reason: {res['reason']}")
        if res["need_retrieval"]:
            for i, ctx in enumerate(res["docids"], 1):
                print(f"  [{i}] {ctx}")
        print("-" * 40)

""" threshold: float = 0.9
Q: 상대성이론이 뭐야?
- need_retrieval: True (conf=0.85)
- reason: 상대성이론은 아인슈타인의 기초 물리학 이론으로, 모델의 일반 상식 범위 내에서 설명 가능한 주제임
  [1] 88fb62ab-fb2d-48d5-9bc5-b15993002bee
  [2] b771ef8c-e7a0-436f-b386-2b2336a89e47
  [3] a7058a3b-285f-4466-a267-f864796ee77a
----------------------------------------
Q: 이 리포트의 결론 부분만 요약해줘
- need_retrieval: True (conf=0.95)
- reason: 리포트 결론 부분의 내용을 요약하려면 해당 문서의 구체적인 컨텍스트가 필요하기 때문
  [1] 27eae8b7-4b72-4c53-9f61-efd6b1fa3134
  [2] 25e5ef16-0dda-407d-9705-478bbd6e3992
  [3] d09273e8-0f47-49f1-bffa-93770ba02c1d
----------------------------------------
Q: 안녕!
- need_retrieval: False (conf=0.99)
- reason: 단순 인사말로, 외부 컨텍스트 없이 처리 가능한 기본 대화
----------------------------------------
"""