import os
import json
from typing import List, Dict, Any

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
from langchain.chains import LLMChain

# 1) Condense 프롬프트
CONDENSE_TEMPLATE = (
    "Given the conversation and a follow-up question, write a single, natural, standalone question in Korean.\n"
    "- Output only one sentence.\n"
    "- Do not include alternatives, parentheses, explanations, or additional lines.\n"
    "- No code blocks, no bullet points, no prefixes.\n\n"
    "Conversation:\n{chat_history}\n\n"
    "Follow-up question: {question}\n\n"
    "Return JSON only:\n{{\"question\": \"...\"}}"
)
prompt = PromptTemplate.from_template(CONDENSE_TEMPLATE)
schema = [ResponseSchema(name="question", description="한 문장 standalone question")]
parser = StructuredOutputParser.from_response_schemas(schema)

os.environ["UPSTAGE_API_KEY"] = ""

# 2) LLM
llm = ChatUpstage(model_name="solar-pro2", temperature=0)

# 3) 히스토리 포맷터
def format_history(messages: List[Dict[str, str]]) -> str:
    lines = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        prefix = "Human" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)

# 4) 한 레코드 처리
def to_standalone(msgs: List[Dict[str, str]]) -> str:
    # msg가 하나면 그대로 질문 반환
    if len(msgs) == 1 and msgs[0].get("role") == "user":
        return msgs[0].get("content", "")
    # 두 개 이상이면 마지막 user를 후속 질문으로 보고 결합
    # 마지막 user 찾기
    last_user = None
    for m in reversed(msgs):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    if not last_user:
        # user 발화가 없으면 빈 문자열
        return ""
    # 대화 히스토리(마지막 user 제외)
    history = []
    seen_last = False
    for m in reversed(msgs):
        if not seen_last and m.get("role") == "user" and m.get("content") == last_user:
            seen_last = True
            continue
        history.append(m)
    history.reverse()
    chat_history_str = format_history(history)

    chain = prompt | llm | parser

    out = chain.invoke({"chat_history": chat_history_str, "question": last_user})
    return out.get("question", "").strip()

# 5) 파일 처리
def main(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            msgs = rec.get("msg", [])
            standalone = to_standalone(msgs)
            # 원본 dict에 standalone_question 추가
            rec["standalone_question"] = standalone
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    os.environ["UPSTAGE_API_KEY"] = ""
    main("/data/ephemeral/home/dev/IR/data/eval.jsonl",
         "/data/ephemeral/home/dev/IR/data/eval_standalone.jsonl")
