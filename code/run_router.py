# run_router.py
import os
import json

from langchain_upstage import ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from router import build_router

def main(input_path: str, output_path: str):
    os.environ["UPSTAGE_API_KEY"] = ""

    llm = ChatUpstage(model_name="solar-pro2", temperature=0)
    router = build_router(llm)

    with (
        open(input_path, "r", encoding="utf-8") as f_in, 
        open(output_path, "w", encoding="utf-8") as f_out
    ):
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)  # 각 줄을 dict로 파싱[1]
            question = rec.get("standalone_question", None)

            routed = router.invoke({"question": question})  # {"intent":..., "confidence":..., "reason":...}[2]
            intent = routed.get("intent", "retrieve")
            reason = routed.get("reason", "")
            try:
                confidence = float(routed.get("confidence", 0.0))
            except Exception:
                confidence = 0.0

            # 기존 딕셔너리에 키 추가
            rec["intent"] = intent
            rec["reason"] = reason
            rec["confidence"] = confidence

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")  # JSONL로 한 줄씩 저장[1]

if __name__ == "__main__":
    os.environ["UPSTAGE_API_KEY"] = ""
    main(
        "/data/ephemeral/home/dev/IR/data/eval_standalone.jsonl",
        "/data/ephemeral/home/dev/IR/data/eval_with_intent.jsonl",
    )
