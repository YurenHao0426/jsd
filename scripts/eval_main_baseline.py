#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline main-task eval: MATH (EM, greedy) and PPL (LM perplexity).
Outputs:
  runs/<DATE>/baseline_eval/main/{math,ppl}/metrics.json
  runs/<DATE>/baseline_eval/main/{math,ppl}/preds.jsonl
"""
import argparse, json, os, math, re, time, pathlib, statistics
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_json(path: str, obj):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_jsonl(path: str, rows: List[Dict]):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def now_ts() -> str:
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# ---------- PPL ----------
@torch.no_grad()
def sequence_nll(model, tok, text: str, device: torch.device) -> float:
    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits  # [1, T, V]
    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = input_ids[:, 1:]
    nll = -(logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).sum().item()
    return float(nll), int(tgt.numel())

def eval_ppl(model, tok, rows: List[Dict], device, outdir: str):
    nll_sum = 0.0
    tok_count = 0
    preds = []
    for r in rows:
        nll, n = sequence_nll(model, tok, r["text"], device)
        nll_sum += nll; tok_count += n
        preds.append({**r, "nll": nll, "tokens": n})
    ppl = math.exp(nll_sum / max(1, tok_count))
    write_json(os.path.join(outdir, "ppl", "metrics.json"), {
        "timestamp": now_ts(), "count": len(rows), "tokens": tok_count, "ppl": ppl
    })
    write_jsonl(os.path.join(outdir, "ppl", "preds.jsonl"), preds)

# ---------- MATH ----------
def canon_num(s: str) -> str:
    # Extract the last integer/decimal (simple heuristic for our 5 examples)
    # Remove commas/whitespace; keep leading minus; allow ^digits not needed here
    s = s.strip()
    # pick last number-like pattern
    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return nums[-1] if nums else s.strip().lower()

@torch.no_grad()
def greedy_generate(model, tok, prompt: str, device, max_new_tokens: int) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        do_sample=False, temperature=0.0, top_p=1.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    # return only the newly generated tail (after prompt)
    prompt_text = tok.decode(enc.input_ids[0], skip_special_tokens=True)
    if text.startswith(prompt_text):
        return text[len(prompt_text):].strip()
    return text.strip()

def eval_math(model, tok, rows: List[Dict], device, outdir: str, max_new_tokens: int):
    correct = 0
    preds = []
    for r in rows:
        q = r["question"]; gold = r["gold"]
        gen = greedy_generate(model, tok, q, device, max_new_tokens=max_new_tokens)
        pred = canon_num(gen); gold_c = canon_num(gold)
        is_ok = int(pred == gold_c)
        correct += is_ok
        preds.append({**r, "gen": gen, "pred": pred, "gold_canon": gold_c, "correct": is_ok})
    acc = correct / max(1,len(rows))
    sd = math.sqrt(acc*(1-acc)/max(1,len(rows)))
    ci = 1.96 * sd
    write_json(os.path.join(outdir, "math", "metrics.json"), {
        "timestamp": now_ts(), "count": len(rows),
        "acc": acc, "acc_ci95": ci
    })
    write_jsonl(os.path.join(outdir, "math", "preds.jsonl"), preds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--math", type=str, required=True)
    ap.add_argument("--ppl", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--max_new_tokens_math", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float16":torch.float16, "bfloat16":torch.bfloat16, "float32":torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if device.type=="cuda" else torch.float32,
        device_map=None
    ).to(device)
    model.eval()

    # MATH
    math_rows = read_jsonl(args.math)
    eval_math(model, tok, math_rows, device, args.out, max_new_tokens=args.max_new_tokens_math)

    # PPL
    ppl_rows = read_jsonl(args.ppl)
    eval_ppl(model, tok, ppl_rows, device, args.out)

    print("[DONE] Main baseline written to", args.out)

if __name__ == "__main__":
    main()
