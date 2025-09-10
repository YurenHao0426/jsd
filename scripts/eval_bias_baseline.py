#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline bias evaluation: CTF (x vs swap(x)), CrowS-Pairs (gender), WinoGender.
- No training; pure scoring with Hugging Face Transformers.
- T = 0 decoding policy: we don't sample; we compute log-probs directly.
Outputs:
  runs/<DATE>/baseline_eval/bias/{ctf,crows,wino}/metrics.json
  runs/<DATE>/baseline_eval/bias/{ctf,crows,wino}/preds.jsonl
"""
import argparse, json, os, math, re, time, pathlib, statistics
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------- IO utils ---------------------
def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_json(path: str, obj: Dict):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_jsonl(path: str, rows: List[Dict]):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# --------------------- Token set mapping ---------------------
def load_word_list(path: str) -> List[str]:
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.append(w)
    return words

def map_words_to_token_ids(tok: AutoTokenizer, words: List[str]) -> List[int]:
    """
    Map words to token ids as single-token variants if possible.
    Try with and without leading space; if both single-tokenize, include both.
    Fall back: if tokenizes to multiple tokens, include the FIRST token id
    (approximation for group-mass aggregation).
    """
    ids = set()
    for w in words:
        cand = []
        for form in (w, " " + w):
            enc = tok(form, add_special_tokens=False, return_tensors=None)
            if len(enc["input_ids"]) == 1:
                cand.append(enc["input_ids"][0])
            else:
                cand.append(enc["input_ids"][0])  # first-piece fallback
        for i in cand:
            ids.add(int(i))
    return sorted(ids)

# --------------------- Scoring utils ---------------------
@torch.no_grad()
def sequence_logprob(model, tok, text: str, device: torch.device) -> float:
    """ Sum log p(y_t | y_<t) over the full sequence (excluding the first token). """
    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits  # [1, T, V]
    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)  # exclude last targetless step
    tgt = input_ids[:, 1:]  # shift
    ll = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
    return float(ll)

@torch.no_grad()
def conditional_logprob(model, tok, prompt: str, cont: str, device: torch.device) -> float:
    """ log p(cont | prompt) by concatenation and subtracting prefix part. """
    e_prompt = tok(prompt, return_tensors="pt", add_special_tokens=False)
    e_cont = tok(" " + cont, return_tensors="pt", add_special_tokens=False)
    input_ids = torch.cat([e_prompt.input_ids, e_cont.input_ids], dim=-1).to(device)
    attn_mask = torch.ones_like(input_ids).to(device)

    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits  # [1, T, V]
    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = input_ids[:, 1:]
    ll_all = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    Lp = e_prompt.input_ids.size(-1)
    ll = ll_all[:, Lp:].sum().item()
    return float(ll)

@torch.no_grad()
def next_token_prob(model, tok, prefix: str, device: torch.device) -> torch.Tensor:
    """ p_t(. | prefix) distribution for the next token. """
    enc = tok(prefix, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits[:, -1, :]  # last position
    probs = F.softmax(logits, dim=-1).squeeze(0)  # [V]
    return probs  # 1-D tensor

def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - m.log())).sum().item() + 0.5 * (q * (q.log() - m.log())).sum().item()

def mean_ci95(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, 0.0
    sd = statistics.pstdev(xs)  # pop std (ok for quick CI)
    # Approx CI with normal approx: 1.96 * sd/sqrt(n)
    ci = 1.96 * (sd / math.sqrt(len(xs)))
    return m, ci

# --------------------- CTF evaluation ---------------------
def first_gender_span(text: str, fem_words: List[str], male_words: List[str]) -> Optional[Tuple[int, int, str]]:
    """
    Find first occurrence (case-insensitive) of any gender word; return (start,end,group)
    group in {"F","M"}.
    """
    lower = text.lower()
    # prefer pronouns first (single-token in most tokenizers)
    fem_sorted = sorted(fem_words, key=len)  # short first
    male_sorted = sorted(male_words, key=len)
    for group, vocab in (("F", fem_sorted), ("M", male_sorted)):
        for w in vocab:
            i = lower.find(w)
            if i != -1:
                return i, i + len(w), group
    return None

def eval_ctf(model, tok, ctf_rows: List[Dict], fem_words, male_words,
             fem_ids: List[int], male_ids: List[int], topk:int, device, outdir:str):
    preds = []
    ctf_gaps = []
    jsds = []
    cover = 0
    for r in ctf_rows:
        x = r["x"]; xs = r["x_swap"]
        # choose boundary after the first gender token to probe next-token distribution
        b1 = first_gender_span(x, fem_words, male_words)
        b2 = first_gender_span(xs, fem_words, male_words)
        if not b1 or not b2:
            preds.append({**r, "skip_reason":"no_gender_span"})
            continue
        p1 = next_token_prob(model, tok, x[:b1[1]], device)
        p2 = next_token_prob(model, tok, xs[:b2[1]], device)

        # group masses
        piF1 = float(p1[fem_ids].sum().item()) if fem_ids else 0.0
        piM1 = float(p1[male_ids].sum().item()) if male_ids else 0.0
        gap1 = abs(piF1 - piM1)

        piF2 = float(p2[fem_ids].sum().item()) if fem_ids else 0.0
        piM2 = float(p2[male_ids].sum().item()) if male_ids else 0.0
        gap2 = abs(piF2 - piM2)

        # record per-sample (we report gap on factual x; swap gap optional)
        gap = gap1
        ctf_gaps.append(gap)
        cover += 1

        # swap JSD at the probe step
        j = jsd(p1, p2)
        jsds.append(float(j))

        preds.append({
            **r,
            "probe_index_factual": b1[1],
            "probe_index_swap": b2[1],
            "piF_factual": piF1, "piM_factual": piM1, "gap_factual": gap1,
            "piF_swap": piF2, "piM_swap": piM2, "gap_swap": gap2,
            "jsd_swap": j
        })

    m_gap, ci_gap = mean_ci95(ctf_gaps)
    m_jsd, ci_jsd = mean_ci95(jsds)
    metrics = {
        "timestamp": now_ts(),
        "count": len(ctf_rows),
        "covered": cover,
        "coverage": (cover / max(1,len(ctf_rows))),
        "CTF_gap_mean": m_gap, "CTF_gap_ci95": ci_gap,
        "JSD_swap_mean": m_jsd, "JSD_swap_ci95": ci_jsd,
        "topk": topk
    }
    write_json(os.path.join(outdir, "ctf", "metrics.json"), metrics)
    write_jsonl(os.path.join(outdir, "ctf", "preds.jsonl"), preds)

# --------------------- CrowS-Pairs (gender) ---------------------
def eval_crows(model, tok, rows: List[Dict], device, outdir:str):
    deltas = []
    preds = []
    for r in rows:
        s_st = r["sentence_stereo"]; s_anti = r["sentence_anti"]
        ll_st = sequence_logprob(model, tok, s_st, device)
        ll_anti = sequence_logprob(model, tok, s_anti, device)
        delta = ll_anti - ll_st
        deltas.append(float(delta))
        preds.append({**r, "logp_stereo": ll_st, "logp_anti": ll_anti, "delta": delta})
    m, ci = mean_ci95(deltas)
    metrics = {
        "timestamp": now_ts(),
        "count": len(rows),
        "delta_logP_mean": m, "delta_logP_ci95": ci
    }
    write_json(os.path.join(outdir, "crows", "metrics.json"), metrics)
    write_jsonl(os.path.join(outdir, "crows", "preds.jsonl"), preds)

# --------------------- WinoGender ---------------------
def eval_wino(model, tok, rows: List[Dict], device, outdir:str):
    correct = 0
    preds = []
    for r in rows:
        prompt = r["premise"]
        a = r["option_a"]; b = r["option_b"]; gold = r["gold"].strip().lower()
        lpa = conditional_logprob(model, tok, prompt, a, device)
        lpb = conditional_logprob(model, tok, prompt, b, device)
        pred = "a" if lpa > lpb else "b"
        correct += int(pred == gold)
        preds.append({**r, "logp_a": lpa, "logp_b": lpb, "pred": pred, "correct": int(pred==gold)})

    acc = correct / max(1,len(rows))
    # quick CI with normal approx
    sd = math.sqrt(acc*(1-acc)/max(1,len(rows)))
    ci = 1.96 * sd
    metrics = {
        "timestamp": now_ts(),
        "count": len(rows),
        "acc": acc, "acc_ci95": ci
    }
    write_json(os.path.join(outdir, "wino", "metrics.json"), metrics)
    write_jsonl(os.path.join(outdir, "wino", "preds.jsonl"), preds)

# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model id, e.g., Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--ctf", type=str, required=True)
    ap.add_argument("--crows", type=str, required=True)
    ap.add_argument("--wino", type=str, required=True)
    ap.add_argument("--groups_dir", type=str, required=True, help="assets/groups/")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16","bfloat16","float32"])
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

    fem_words = load_word_list(os.path.join(args.groups_dir, "en_female.txt"))
    male_words = load_word_list(os.path.join(args.groups_dir, "en_male.txt"))
    fem_ids = map_words_to_token_ids(tok, fem_words)
    male_ids = map_words_to_token_ids(tok, male_words)

    outdir = args.out

    # CTF
    ctf_rows = read_jsonl(args.ctf)
    eval_ctf(model, tok, ctf_rows, fem_words, male_words, fem_ids, male_ids, args.top_k, device, outdir)

    # CrowS
    crows_rows = read_jsonl(args.crows)
    eval_crows(model, tok, crows_rows, device, outdir)

    # Wino
    wino_rows = read_jsonl(args.wino)
    eval_wino(model, tok, wino_rows, device, outdir)

    print("[DONE] Bias baseline written to", outdir)

if __name__ == "__main__":
    main()
