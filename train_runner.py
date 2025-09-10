#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-step LoRA training runner for EM / Group-EM / JSD.
- Generates a short greedy continuation for each prompt (T=0), then teacher-forces
  on [prompt + continuation] to compute stepwise logits and apply losses.
- Gating: Top-K membership of {F ∪ M} at each step (k configurable).
- Guards: mass parity (lambda) and stability KL to a frozen base model (beta).
Outputs:
  adapter weights + a JSON log of loss components.
This script is single-GPU; to parallelize, launch multiple processes with different CUDA_VISIBLE_DEVICES.
"""
import os, json, math, time, random, pathlib, argparse
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from src.losses import (
    map_words_to_token_ids, probs_from_logits, topk_gate,
    loss_em, loss_group_em, loss_jsd
)

def set_seed(s: int):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def read_jsonl(p: str) -> List[Dict]:
    return [json.loads(x) for x in open(p, "r", encoding="utf-8") if x.strip()]

def save_json(p: str, obj):
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    open(p, "w", encoding="utf-8").write(json.dumps(obj, indent=2))

@torch.no_grad()
def greedy_generate_ids(model, tok, prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """
    prompt_ids: [1, L]
    return gen_ids: [1, G] (without BOS/EOS trimming)
    """
    out = model.generate(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        do_sample=False, temperature=0.0, top_p=1.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id
    )
    # decode original prompt length to slice
    gen_ids = out[:, prompt_ids.size(1):]  # [1, G]
    return gen_ids

def build_lora(model) -> nn.Module:
    lconf = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]  # Qwen-style
    )
    return get_peft_model(model, lconf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--loss_type", type=str, choices=["em","group_em","jsd"], required=True)
    ap.add_argument("--train_file", type=str, help="for em/group_em: JSONL with {'prompt':...}")
    ap.add_argument("--train_pairs", type=str, help="for jsd: JSONL with {'prompt','prompt_swap'}")
    ap.add_argument("--groups_dir", type=str, default="assets/groups")
    ap.add_argument("--output_dir", type=str, required=True)

    # optimization
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--gen_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2025)

    # guards / gating
    ap.add_argument("--lambda_mass", type=float, default=0.0)
    ap.add_argument("--beta_kl", type=float, default=0.0)
    ap.add_argument("--topk_gate", type=int, default=20)
    ap.add_argument("--topk_jsd", type=int, default=0)   # 0 => full vocab JSD

    # dtype / device
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16","float16","float32"])
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # Load tokenizer/model + LoRA (trainable) and base (frozen for KL)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    base_model.eval()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model = build_lora(model)
    model.train()

    # Build optimizer/scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    # Build gender token id sets (for group and jsd; EM ignores)
    fem_words = [w.strip().lower() for w in open(os.path.join(args.groups_dir,"en_female.txt"),"r",encoding="utf-8")]
    male_words = [w.strip().lower() for w in open(os.path.join(args.groups_dir,"en_male.txt"),"r",encoding="utf-8")]
    fem_ids = map_words_to_token_ids(tok, fem_words)
    male_ids = map_words_to_token_ids(tok, male_words)

    # Load training data
    if args.loss_type in ("em","group_em"):
        assert args.train_file, "--train_file is required for em/group_em"
        rows = read_jsonl(args.train_file)
        assert len(rows) > 0, "Empty train_file"
        prompts = [r["prompt"] for r in rows]
    else:
        assert args.loss_type == "jsd" and args.train_pairs, "--train_pairs is required for jsd"
        rows = read_jsonl(args.train_pairs)
        assert len(rows) > 0, "Empty train_pairs"
        pairs = [(r["prompt"], r["prompt_swap"]) for r in rows]

    # Logging
    outdir = pathlib.Path(args.output_dir)
    (outdir/"logs").mkdir(parents=True, exist_ok=True)
    (outdir/"adapter").mkdir(parents=True, exist_ok=True)
    json_log = []

    step = 0
    while step < args.max_steps:
        # simple cyclic sampler
        if args.loss_type in ("em","group_em"):
            idx = step % len(prompts)
            prompt = prompts[idx]
            # 1) generate a short continuation (greedy)
            enc = tok(prompt, return_tensors="pt")
            gen_ids = greedy_generate_ids(model, tok, enc.input_ids.to(device), max_new_tokens=args.gen_len)  # [1,G]
            concat_ids = torch.cat([enc.input_ids.to(device), gen_ids], dim=1)                                # [1,L+G]
            attn = torch.ones_like(concat_ids).to(device)

            # 2) forward current model (teacher forcing) for loss
            out = model(input_ids=concat_ids, attention_mask=attn)
            logits = out.logits[:, :-1, :]     # predict next for each pos except last targetless
            T = logits.size(1)
            # generation mask: 0 for prompt part, 1 for generated part (shifted by 1)
            gen_mask = torch.zeros((1,T), dtype=torch.float32, device=device)
            gen_mask[:, enc.input_ids.size(1)-1:] = 1.0  # from the last prompt position onward

            if args.loss_type == "em":
                loss, extras = loss_em(logits, gen_mask)
            else:
                gate = topk_gate(logits, fem_ids, male_ids, k=args.topk_gate)  # [1,T]
                with torch.no_grad():
                    out_ref = base_model(input_ids=concat_ids, attention_mask=attn)
                    ref_probs = probs_from_logits(out_ref.logits[:, :-1, :])
                loss, extras = loss_group_em(
                    logits, gen_mask, fem_ids, male_ids, gate_mask=gate,
                    lambda_mass=args.lambda_mass, beta_kl=args.beta_kl, ref_probs=ref_probs
                )

        else:
            # JSD branch
            idx = step % len(pairs)
            x, xs = pairs[idx]

            # factual
            enc_f = tok(x, return_tensors="pt")
            gen_f = greedy_generate_ids(model, tok, enc_f.input_ids.to(device), max_new_tokens=args.gen_len)
            all_f = torch.cat([enc_f.input_ids.to(device), gen_f], dim=1)
            attn_f = torch.ones_like(all_f).to(device)
            out_f = model(input_ids=all_f, attention_mask=attn_f)
            logits_f = out_f.logits[:, :-1, :]
            T_f = logits_f.size(1)
            gen_mask_f = torch.zeros((1,T_f), dtype=torch.float32, device=device)
            gen_mask_f[:, enc_f.input_ids.size(1)-1:] = 1.0
            gate_f = topk_gate(logits_f, fem_ids, male_ids, k=args.topk_gate)

            # counterfactual
            enc_c = tok(xs, return_tensors="pt")
            gen_c = greedy_generate_ids(model, tok, enc_c.input_ids.to(device), max_new_tokens=args.gen_len)
            all_c = torch.cat([enc_c.input_ids.to(device), gen_c], dim=1)
            attn_c = torch.ones_like(all_c).to(device)
            out_c = model(input_ids=all_c, attention_mask=attn_c)
            logits_c = out_c.logits[:, :-1, :]

            with torch.no_grad():
                out_ref_f = base_model(input_ids=all_f, attention_mask=attn_f)
                ref_probs_f = probs_from_logits(out_ref_f.logits[:, :-1, :])

            loss, extras = loss_jsd(
                logits_f, logits_c, gen_mask_f, fem_ids, male_ids,
                gate_mask_f=gate_f, lambda_mass=args.lambda_mass, beta_kl=args.beta_kl,
                ref_probs_f=ref_probs_f, topk_jsd=args.topk_jsd
            )

        (loss / args.grad_accum).backward()
        if (step + 1) % args.grad_accum == 0 or (step + 1) == args.max_steps:
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)

        step += 1
        json_log.append({"step": step, "loss": float(loss.item()), **extras})
        if step % 1 == 0:
            print(f"[{step}/{args.max_steps}] loss={loss.item():.6f} | {extras}")

    # save adapter
    model.save_pretrained(str(outdir/"adapter"))
    save_json(str(outdir/"logs"/"train_log.json"), {"args": vars(args), "log": json_log})
    print("Saved adapter to", outdir/"adapter")

if __name__ == "__main__":
    main()
