import os
from dotenv import load_dotenv

load_dotenv()

from src.llm.gateway import LLMGateway
from src.rag.orchestrator import MODEL_CLASSIFY, MODEL_QUERY_TRANSFORM, MODEL_SIMPLE, MODEL_COMPLEX
from src.rag.confidence import MODEL_EVAL

def test_model_chain(name, chain, max_tokens=20, reasoning_effort=None):
    gateway = LLMGateway()
    print(f"\n--- Testing {name} ---")
    print(f"Chain: {chain}")
    try:
        res = gateway.generate(
            prompt="Xin chào, hãy trả lời ngắn gọn: 'Test OK'",
            model=chain,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        print(f"✅ Success! Response: {res}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    print("Testing Model Chains...")
    # reasoning_effort/max_tokens below mirror each chain's real call site in
    # orchestrator.py/confidence.py — testing with different params than
    # production uses can mask a real failure (or report a fake one) for
    # thinking-capable models (Gemini, gpt-oss).
    test_model_chain("MODEL_CLASSIFY", MODEL_CLASSIFY, max_tokens=150, reasoning_effort="none")
    test_model_chain("MODEL_QUERY_TRANSFORM", MODEL_QUERY_TRANSFORM, max_tokens=250, reasoning_effort="none")
    test_model_chain("MODEL_SIMPLE", MODEL_SIMPLE, max_tokens=250)
    test_model_chain("MODEL_COMPLEX", MODEL_COMPLEX, max_tokens=250)
    test_model_chain("MODEL_EVAL", MODEL_EVAL, max_tokens=250, reasoning_effort="none")
