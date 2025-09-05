# One-shot Entropy Minimization

[![paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.20282)
[![Model](https://img.shields.io/badge/Models/Dataset-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/zgao3186/qwen25math7b-one-shot-em/)
[![Notion](https://img.shields.io/badge/Site-000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://www.notion.so/One-shot-Entropy-Minimization-202606db813b80639773f850f39246a5) 

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
