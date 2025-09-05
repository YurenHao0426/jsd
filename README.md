# One-shot Entropy Minimization

### Installation

```bash
conda create -n one-shot-em python=3.10 -y
pip install -r requirements.txt
```

---

### Reproducing One-shot EM Training (SOTA)

```bash
accelerate launch train.py \
  --model_name Qwen2.5-Math-7B \
  --model_path /path/to/Qwen2.5-Math-7B \
  --train_data dataset/1shot_rlvr/pi1_r1280.parquet \
  --effective_batch 64 \
  --micro_batch_size 2 \
  --temperature 0.5 \
  --learning_rate 2e-5 \
  --max_steps 50 \
  --log_steps 1 \
  --save_steps 1 \
  --run_name one_shot \
  --wandb_project one-shot-em
```

---

### Reproducing Multi-shot EM Training

```bash
accelerate launch train.py \
  --model_name Qwen2.5-Math-7B \
  --model_path /path/to/Qwen2.5-Math-7B \
  --train_data dataset/numina/numina_00.parquet \
  --effective_batch 64 \
  --micro_batch_size 2 \
  --temperature 0.5 \
  --learning_rate 2e-5 \
  --max_steps 50 \
  --log_steps 1 \
  --save_steps 1 \
  --run_name multi_shot \
  --wandb_project one-shot-em
```

---

### Evaluation

```bash
cd Qwen2.5-Eval/evaluation
bash sh/eval_all_math.sh
```

---

### Caching (Hugging Face)

To avoid repeated downloads across runs, we persist Hugging Face caches in the user cache directory. When activating the `one-shot-em` conda environment, the following environment variables are set:

```bash
HF_HOME="$HOME/.cache/huggingface"
HF_DATASETS_CACHE="$HF_HOME/datasets"
HF_HUB_CACHE="$HF_HOME/hub"
TRANSFORMERS_CACHE="$HF_HUB_CACHE"
```

You can change these by editing the conda env activation hook under:

```
$CONDA_PREFIX/etc/conda/activate.d/98-hf-cache.sh
```

Models and tokenizers are cached under `~/.cache/huggingface/hub` and will be reused automatically.

---

### Weights & Tokenizer Prefetch (Qwen2.5-7B-Instruct)

To pre-download the text-only Instruct variant (not long-context/multimodal) and its tokenizer into the cache:

```bash
conda activate one-shot-em
python - <<'PY'
from huggingface_hub import snapshot_download
repo = "Qwen/Qwen2.5-7B-Instruct"
# First grab tokenizer-related small files (fast verification)
snapshot_download(repo_id=repo, allow_patterns=[
    "tokenizer*","vocab*","merges*",
    "special_tokens_map.json","tokenizer.json",
    "tokenizer_config.json","tokenizer.model",
], resume_download=True)
# Then optionally grab the full snapshot (large download; resumes automatically)
snapshot_download(repo_id=repo, resume_download=True)
PY
```

---

### Accelerate Configuration

We keep a default Accelerate config at:

```
configs/accelerate/default_config.yaml
```

This is a placeholder you can modify with `accelerate config` for multi-GPU runs later.

---

### Weights & Biases (W&B)

By default, W&B is disabled in the `one-shot-em` environment. To enable it, unset `WANDB_DISABLED` (or set it to `false`) and ensure your API key is set, for example:

```bash
export WANDB_DISABLED=false
export WANDB_API_KEY=...  # your key
```

If you wish to keep it off (default), no action is required.

---

### One-click Self-check

Run a comprehensive environment check (cache, model/tokenizer, W&B, Accelerate, GPU) and write a hardware snapshot to `docs/hardware.md`:

```bash
conda activate one-shot-em
python scripts/self_check.py
```

To avoid writing the hardware snapshot (stdout only):

```bash
python scripts/self_check.py --no-write
```

The script will also verify that `Qwen/Qwen2.5-7B-Instruct` (text-only Instruct) is cached and loadable locally.

---

### Acknowledgements

Our dataset references and builds upon the following open-source contributions:

- [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- [DeepScaler](https://github.com/agentica-project/deepscaler)
- [One-shot RLVR](https://github.com/ypwang61/One-Shot-RLVR/) – for data selection strategies
- [Qwen2.5-Eval](https://github.com/QwenLM/Qwen2.5-Math/) – for evaluation benchmarks

We sincerely thank the authors and maintainers of these projects for their excellent contributions to the research community!


---

### Citation
```
@misc{gao2025oneshotentropyminimization,
      title={One-shot Entropy Minimization}, 
      author={Zitian Gao and Lynx Chen and Haoming Luo and Joey Zhou and Bryan Dai},
      year={2025},
      eprint={2505.20282},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20282}, 
}
```
