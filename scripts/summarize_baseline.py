#!/usr/bin/env python3
import json, os, sys, pathlib, datetime as dt

def loadj(p): 
    try:
        with open(p,'r',encoding='utf-8') as f: return json.load(f)
    except: return None

def main(root):
    root = pathlib.Path(root)
    out = root/"summary.md"
    bias_ctf = loadj(root/"bias/ctf/metrics.json")
    bias_crows = loadj(root/"bias/crows/metrics.json")
    bias_wino = loadj(root/"bias/wino/metrics.json")
    main_math = loadj(root/"main/math/metrics.json")
    main_ppl = loadj(root/"main/ppl/metrics.json")

    lines = ["# Baseline Summary",
             f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
             "","## Bias"]
    if bias_ctf:
        lines.append(f"- **CTF-gap**: {bias_ctf['CTF_gap_mean']:.6f} ± {bias_ctf['CTF_gap_ci95']:.6f} (coverage={bias_ctf['coverage']:.2f})")
        lines.append(f"- **JSD_swap**: {bias_ctf['JSD_swap_mean']:.6f} ± {bias_ctf['JSD_swap_ci95']:.6f}")
    if bias_crows:
        lines.append(f"- **CrowS ΔlogP** (anti−stereo): {bias_crows['delta_logP_mean']:.6f} ± {bias_crows['delta_logP_ci95']:.6f}")
    if bias_wino:
        lines.append(f"- **Wino Acc**: {bias_wino['acc']:.3f} ± {bias_wino['acc_ci95']:.3f}")
    lines += ["","## Main"]
    if main_math:
        lines.append(f"- **MATH EM**: {main_math['acc']:.3f} ± {main_math['acc_ci95']:.3f}")
    if main_ppl:
        lines.append(f"- **PPL**: {main_ppl['ppl']:.2f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines)+"\n",encoding='utf-8')
    print("Wrote", out)

if __name__=="__main__":
    # usage: python scripts/summarize_baseline.py runs/20250910/baseline_eval
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
