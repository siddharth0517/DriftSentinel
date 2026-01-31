import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------
# ENV + CLIENT SETUP
# -------------------------------------------------
load_dotenv()

API_KEY = os.getenv("nv")
if not API_KEY:
    raise RuntimeError("OPENROUTER API KEY NOT FOUND")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b:free"


# -------------------------------------------------
# UTILITY: NUMPY → PYTHON TYPES
# -------------------------------------------------
def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    else:
        return obj


# -------------------------------------------------
# STREAMING DRIFT EXPLAINER WITH REASONING
# -------------------------------------------------
def stream_drift_explanation(drift_report: dict):
    """
    Streams a concise AI explanation for detected drift.
    Reasoning is enabled but NOT exposed directly.
    """

    clean_report = to_python_types(drift_report)

    # ---- SYSTEM PROMPT (IMPORTANT) ----
    prompt = f"""
You are a Senior Data Scientist responsible for monitoring ML systems in production.

Drift Report:
{json.dumps(clean_report, indent=2)}

Your task:
1. Identify the drifting feature.
2. Explain the real-world business risk.
3. Recommend ONE immediate action.

Rules:
- Be concise and factual.
- Do NOT mention statistics explicitly unless needed.
- Do NOT include chain-of-thought or internal reasoning.
"""

    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True,
            extra_body={
                "reasoning": {
                    "enabled": True   # model reasons internally
                }
            }
        )

        # Stream only visible content (NOT reasoning details)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except Exception as e:
        yield f"⚠️ AI Explanation Error: {str(e)}"
