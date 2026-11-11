"""Inference for the Qwen2.5-1.5B-Instruct NSS LoRA adapter.

Examples:
    python infer_nss.py --narrative "Growth weakens and inflation falls."
    python infer_nss.py --narrative-file scenario.txt --today-json today.json --blend 0.6
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import torch
from jsonschema import validate
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

KEYS = ["b0", "b1", "b2", "b3", "t1", "t2"]
SYSTEM = (
    "You map a Swiss macro-financial narrative to Nelson-Siegel-Svensson parameters. "
    "Return only one valid JSON object with exactly the numeric keys b0, b1, b2, b3, t1 and t2. "
    "The decay parameters t1 and t2 must be strictly positive. Do not add prose or Markdown."
)
SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "number"} for k in KEYS},
    "required": KEYS,
    "additionalProperties": False,
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--narrative")
    src.add_argument("--narrative-file")
    p.add_argument("--model-root", default="models")
    p.add_argument("--today-json", help="Optional JSON file containing today's six NSS parameters.")
    p.add_argument("--blend", type=float, default=1.0, help="0=today, 1=model scenario; default 1.")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    return p.parse_args()


def extract_json(text: str) -> Dict[str, float]:
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.S)
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
            validate(instance=obj, schema=SCHEMA)
            out = {k: float(obj[k]) for k in KEYS}
            if out["t1"] <= 0 or out["t2"] <= 0:
                raise ValueError("t1 and t2 must be positive")
            return out
        except Exception:
            continue
    raise ValueError(f"No valid NSS JSON found in model output: {text!r}")


def load_today(path: str) -> Dict[str, float]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(instance=obj, schema=SCHEMA)
    return {k: float(obj[k]) for k in KEYS}


def main() -> None:
    a = args()
    if not 0.0 <= a.blend <= 1.0:
        raise ValueError("--blend must be between 0 and 1")

    narrative = a.narrative or Path(a.narrative_file).read_text(encoding="utf-8")
    root = Path(a.model_root)
    base_dir = root / "base_qwen25_1_5b"
    adapter_dir = root / "nss_qwen25_1_5b_lora"
    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": narrative.strip()},
    ]
    # 1. Apply Qwen's chat template
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 2. Tokenize the resulting prompt
    model_inputs = tokenizer(
        [prompt_text],
        return_tensors="pt"
    )

    # 3. Generate
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=a.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 4. Remove the original prompt tokens
    prompt_length = model_inputs["input_ids"].shape[1]
    answer_tokens = generated[0, prompt_length:]

    # 5. Decode only the newly generated response
    raw = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True
    ).strip()
    raw = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    scenario = extract_json(raw)

    if a.today_json:
        today = load_today(a.today_json)
        scenario = {k: today[k] + a.blend * (scenario[k] - today[k]) for k in KEYS}

    print(json.dumps(scenario, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
