import argparse
import os
import random
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import wandb

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


os.environ.setdefault("NCCL_TIMEOUT", "2700")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "2700")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen2.5-Math-7B', help='Model name')
    parser.add_argument('--model_path', type=str, default=None, help='Local model path')
    parser.add_argument('--train_data', type=str, default='dataset/1shot_rlvr/pi1_r1280.parquet', help='Training data file path')
    parser.add_argument('--save_root', type=str, default=None, help='Checkpoint save root directory')
    parser.add_argument('--effective_batch', type=int, default=64, help='Global batch size')
    parser.add_argument('--micro_batch_size', type=str, default=2, help='Micro batch size or "auto"')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature coefficient')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--log_steps', type=int, default=1, help='Logging step interval')
    parser.add_argument('--save_steps', type=int, default=1, help='Checkpoint saving step interval')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum training steps')
    parser.add_argument('--sample_temp', type=float, default=0.5, help='Generation temperature parameter')
    parser.add_argument('--run_name', type=str, default=None, help='Experiment run name')
    parser.add_argument('--wandb_project', type=str, default='entropy-maximization-ft', help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--seed', type=int, default=15, help='Random seed')
    return parser.parse_args()

class FTDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

def custom_collate(batch):
    return {"input": [item["input"] for item in batch]}

def get_optimal_micro_batch_size(model_name: str, world_size: int = 1) -> int:
    model_configs = {
        "1.5B": {"base_batch": 4, "keywords": ["1.5B", "1B"]},
        "2B": {"base_batch": 4, "keywords": ["2B"]},
        "3B": {"base_batch": 2, "keywords": ["3B"]},
        "7B": {"base_batch": 2, "keywords": ["7B"]},
        "8B+": {"base_batch": 1, "keywords": ["8B", "9B", "10B", "11B", "12B", "13B", "14B"]},
    }
    model_name_upper = model_name.upper()
    detected = next((cfg for cfg in model_configs.values() if any(k in model_name_upper for k in cfg["keywords"])), None)
    base_batch = detected["base_batch"] if detected else 2
    if world_size > 1:
        return min(base_batch + 1, int(base_batch * 1.5))
    return base_batch

def apply_chat_template(tokenizer, problem: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False, add_generation_prompt=True
    )

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    micro_bs = int(args.micro_batch_size)
    eff_bs = args.effective_batch
    accum_steps = max(1, eff_bs // (micro_bs * world_size))
    temp = args.temperature
    lr = args.learning_rate

    save_root = args.save_root or (f"checkpoints/{args.model_name}/{args.run_name}" if args.run_name else f"checkpoints/{args.model_name}")
    ds_config = {
        "train_micro_batch_size_per_gpu": micro_bs,
        "train_batch_size": eff_bs,
        "gradient_accumulation_steps": accum_steps,
        "bf16": {"enabled": True},
        "zero_optimization": {
                              "stage": 2, 
                              "offload_optimizer": {"device": "cpu"}, 
                              "offload_param": {"device": "none"}
                             },
        "gradient_clipping": 1.0,
    }
    accelerator = Accelerator(mixed_precision="bf16", 
                              gradient_accumulation_steps=accum_steps, 
                              deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds_config))
    print = accelerator.print

    model_path = args.model_path or f"/volume/pt-train/models/{args.model_name}"
    config = AutoConfig.from_pretrained(model_path)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.run_name or args.wandb_name or args.model_name, config=vars(args))

    df = pd.read_parquet(args.train_data)
    train_data = [{"input": apply_chat_template(tokenizer, p)} for p in df["problem"].dropna().tolist()]
    train_loader = DataLoader(FTDataset(train_data), batch_size=micro_bs, shuffle=True, collate_fn=custom_collate)

    optimizer = AdamW(model.parameters(), lr=lr)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    prev_logits = None
    model.train()
    
    for step, batch in enumerate(train_loader, start=1):
        if step > args.max_steps:
            print(f"Exceed max step {args.max_steps}, training stopped.")
            break
        
        with accelerator.accumulate(model):
            enc = tokenizer(batch["input"], 
                            return_tensors="pt", 
                            padding="longest", 
                            truncation=True, 
                            max_length=2048).to(accelerator.device)
            
            with torch.no_grad():
                gen_ids = accelerator.unwrap_model(model).generate(**enc, 
                                                                   max_new_tokens=512, 
                                                                   do_sample=True, 
                                                                   top_p=0.95, 
                                                                   temperature=args.sample_temp, 
                                                                   synced_gpus=True, 
                                                                   repetition_penalty=1.15,
                                                                   pad_token_id=tokenizer.pad_token_id, 
                                                                   use_cache=False)
                
            seq = torch.cat([enc.input_ids, gen_ids[:, enc.input_ids.shape[1]:]], dim=1)[:, :4096]
            pad_mask = seq.ne(tokenizer.pad_token_id)
            prompt_len = pad_mask[:, :enc.input_ids.shape[1]].sum(-1)
            token_idx = torch.arange(seq.size(1), device=seq.device)
            gen_mask = (token_idx.unsqueeze(0) >= prompt_len.unsqueeze(1)) & pad_mask

            logits = model(seq, attention_mask=pad_mask).logits
            probs = F.softmax(logits / temp, dim=-1)
            H_tok = -(probs * torch.log(probs + 1e-12)).sum(-1)
            loss = (H_tok * gen_mask).sum() / gen_mask.sum().clamp_min(1)

            prev_logits = logits.detach()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.is_main_process:
            if step % args.log_steps == 0:
                print(f"Step {step} | loss={loss.item():.6f}")
                wandb.log({"step": step, "loss": loss.item()})
                
            if step % args.save_steps == 0:
                ckpt = Path(save_root) / f"step_{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(ckpt, safe_serialization=True)
                tokenizer.save_pretrained(ckpt)
                print(f"Checkpoint saved to {ckpt}")

    if accelerator.is_main_process:
        final = Path(save_root) / "final"
        final.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final, safe_serialization=True)
        tokenizer.save_pretrained(final)
        print(f"Final checkpoint saved to {final}")
        wandb.finish()

if __name__ == "__main__":
    main()
