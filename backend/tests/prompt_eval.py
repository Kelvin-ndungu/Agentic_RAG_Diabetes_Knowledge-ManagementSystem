"""Lightweight prompt regression harness."""
import json
import os
from pathlib import Path

from backend.rag import classify_query_unified
from backend.schema import ChatState
from langchain_core.messages import HumanMessage


def main():
    cases_path = Path(__file__).parent / "prompt_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    run_llm = os.getenv("RUN_LLM", "0") == "1"
    # Justification: default dry-run avoids accidental API usage and cost.
    if not run_llm:
        print("RUN_LLM not set; printing cases only.")
        for case in cases:
            print(f"- {case['id']}: {case['message']}")
        return

    for case in cases:
        state = ChatState(messages=[HumanMessage(content=case["message"])])
        result_state = classify_query_unified(state)
        output = result_state.get("classifier_output")
        print(f"[{case['id']}] route={output.route} safety={output.safety} intent={output.intent}")


if __name__ == "__main__":
    main()
