"""Mini live-check for every LLM provider key in .env.

Sends one tiny completion per (model, key) pair — bypassing the gateway's
fallback chain so a broken key can't hide behind a healthy provider.
Costs a handful of requests; safe to run whenever keys change.

Usage:
    python scripts/check_llm_keys.py            # every model in MODEL_CHAIN
    python scripts/check_llm_keys.py --fast     # one model per provider
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
from dotenv import load_dotenv

load_dotenv()

import litellm  # noqa: E402

from src.llm.gateway import LLMGateway  # noqa: E402

PROMPT = "2+2=? Trả lời ngắn gọn."


@click.command()
@click.option("--fast", is_flag=True, default=False,
              help="Chỉ test model đầu tiên của mỗi provider (ít request hơn).")
def main(fast: bool):
    gw = LLMGateway()
    models = gw._usable_models(list(gw.MODEL_CHAIN))
    if not models:
        click.echo("⚠️ Không có provider nào có key trong .env.")
        sys.exit(1)
    if fast:
        seen = set()
        models = [m for m in models
                  if gw._provider(m) not in seen and not seen.add(gw._provider(m))]

    n_fail = 0
    for model in models:
        for key in gw._keys_for(model):
            label = f"{model}  key(...{key[-4:]})"
            kwargs = {}
            if gw._is_thinking_model(model):
                kwargs["reasoning_effort"] = "none"
            try:
                r = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": PROMPT}],
                    max_tokens=100, temperature=0, api_key=key, **kwargs,
                )
                content = (r.choices[0].message.content or "").strip()
                content = " ".join(content.split())[:60]
                click.echo(f"✅ OK    {label}: {content!r}")
            except Exception as e:
                n_fail += 1
                msg = " ".join(str(e).split())[:180]
                click.echo(f"❌ FAIL  {label}: {type(e).__name__}: {msg}")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
