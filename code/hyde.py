import os
import json
from typing import List, Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage

# 1) Condense 프롬프트
CONDENSE_TEMPLATE = ("""
주어진 콘텐츠를 읽고 관련된 간단한 질문 3개를 만든다.

제약:
- 각 질문은 “?”로 끝나는 한 문장.
- 중복/유사 금지, 콘텐츠 범위 내에서만 작성.
- 괄호/선택지/설명/접두어/불릿/코드블록 금지.
- 각 질문은 50자 이내.
- 콘텐츠가 비어있거나 의미가 없으면 빈 배열을 반환.

오직 JSON으로만 출력:
{{
  "questions": ["...", "...", "..."]
}}

콘텐츠:
{content}
"""
)
prompt = PromptTemplate.from_template(CONDENSE_TEMPLATE)
parser = JsonOutputParser()

# 1. LLM 으로 각 doc의 쿼리 3개씩 생성하고 저장
def make_questions():
    os.environ["UPSTAGE_API_KEY"] = ""
    llm = ChatUpstage(model_name="solar-pro2", temperature=0)
    chain = prompt | llm | parser

    inpath = '/data/ephemeral/home/dev/IR/data/documents.jsonl'
    outpath = '/data/ephemeral/home/dev/IR/data/documents_question.jsonl'

    with (
        open (inpath, "r", encoding="utf-8") as f_in,
        open (outpath, "w", encoding="utf-8") as f_out
    ):
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            content = rec.get("content", "")
            questions = chain.invoke({"content":content})
            rec["questions"] = questions.get("questions")
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")


# 2. 각 doc의 질문의 쿼리와 사용자의 쿼리를 확률 비교

if __name__ == "__main__":
    make_questions()


