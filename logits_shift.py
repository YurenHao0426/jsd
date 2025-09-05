import os, gc, random
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import skew

PARQUET_PATH = "data/train/one_shot_rlvr/dsr_sub.parquet"
NUM_SAMPLES = 20
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.6
PAD_TOKEN_ID = 151643

model_paths = {
    "A": "/volume/ailab4sci/models/Qwen2.5-Math-7B", # base
    "B": "/volume/ailab4sci/ztgao/em/checkpoints/qwen25math7b/t05_lr2e-05_bsz64_seed15/step_10", # em
    "C": "/volume/ailab4sci/ztgao/One-Shot-RLVR/checkpoints/verl_few_shot/Qwen2.5-Math-7B-origin-dsr_sub/global_step_460/actor", # rl
    "D": "/volume/ailab4sci/ztgao/One-Shot-RLVR/checkpoints/verl_few_shot/Qwen2.5-Math-7B-em_step10/global_step_460/actor" # emrl
}

torch.manual_seed(42)

df = pd.read_parquet(PARQUET_PATH).head(NUM_SAMPLES)
prompts = df["prompt"].tolist()
messages_list = [list(arr) for arr in prompts]
logits_dict = {m: [] for m in model_paths}

for model_name, model_path in model_paths.items():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    for msg in tqdm(messages_list, desc=f"{model_name} samples"):
        text = tokenizer.apply_chat_template(msg, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=PAD_TOKEN_ID,
            )

        logits_steps = [score[0].cpu().numpy() for score in output.scores]
        logits_dict[model_name].append(logits_steps)

    del model
    gc.collect()
    torch.cuda.empty_cache()

min_len = min(
    len(step_list) for model in logits_dict.values() for step_list in model
)

for model in logits_dict:
    logits_dict[model] = [steps[:min_len] for steps in logits_dict[model]]

def flatten_model_logits(model_logits):
    flat = np.concatenate([np.stack(step_list).reshape(-1) for step_list in model_logits])
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return flat

flat_A = flatten_model_logits(logits_dict["A"])
flat_B = flatten_model_logits(logits_dict["B"])
flat_C = flatten_model_logits(logits_dict["C"])
flat_D = flatten_model_logits(logits_dict["D"])

def stable_softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

prob_A = stable_softmax(flat_A)
prob_B = stable_softmax(flat_B)
prob_C = stable_softmax(flat_C)
prob_D = stable_softmax(flat_D)


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
colors = ["#6BAED6", "#9ECAE1", "#C6DBEF", "#08306B"]

all_logits = []
flattened_logits_dict = {}

for name, logits_list in logits_dict.items():
    logits_all = torch.tensor(np.stack(logits_list))
    logits_all = torch.nan_to_num(logits_all, nan=0.0, posinf=0.0, neginf=0.0)
    flattened = logits_all[logits_all != 0].flatten().cpu().numpy()

    if flattened.size == 0:
        continue

    flattened_logits_dict[name] = flattened
    all_logits.append(flattened)

combined = np.concatenate(all_logits)
global_min = combined.min()
global_max = combined.max()

for i, (name, flat_logits) in enumerate(flattened_logits_dict.items()):
    ax = axes[i]
    ax.hist(flat_logits, bins=50, color=colors[i], alpha=0.8)
    ax.set_xlim(global_min, global_max)
    names = ["Base", "EM", "RL", "EM-RL"]
    ax.set_title(f"{names[i]} Logits")
    ax.set_xlabel("Logit Value")
    ax.set_ylabel("Frequency")
    mu = flat_logits.mean()
    sigma = flat_logits.std()
    skewness = skew(flat_logits)
    stats_text = f"μ={mu:.2f}\nσ={sigma:.2f}\nskew={skewness:.2f}"
    ax.text(0.985, 0.97, stats_text, transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

for j in range(len(flattened_logits_dict), 4):
    axes[j].axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("logits.pdf", format="pdf", bbox_inches="tight")
plt.show()