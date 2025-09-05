#!/usr/bin/env python3
"""
One-click environment self-check for the One-shot EM project.

This script validates:
1) Hugging Face cache variables and directories
2) Presence of Qwen/Qwen2.5-7B-Instruct model and tokenizer in local cache
3) Weights & Biases disablement via environment variables
4) Accelerate configuration placeholder existence
5) GPU visibility via nvidia-smi and PyTorch

It also writes a concise hardware snapshot to docs/hardware.md and prints a
human-readable report to stdout.

Note: All code/comments are kept in English to follow project policy.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
HARDWARE_MD = DOCS_DIR / "hardware.md"
ACCELERATE_CFG = REPO_ROOT / "configs" / "accelerate" / "default_config.yaml"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a command and return (code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError as exc:
        return 127, "", str(exc)


def check_env_vars() -> Dict[str, Optional[str]]:
    keys = [
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "WANDB_DISABLED",
        "WANDB_MODE",
        "WANDB_SILENT",
        "CONDA_PREFIX",
    ]
    return {k: os.environ.get(k) for k in keys}


@dataclasses.dataclass
class HFCaches:
    hf_home: Optional[str]
    datasets_cache: Optional[str]
    hub_cache: Optional[str]
    transformers_cache: Optional[str]


def ensure_dirs(paths: List[str]) -> List[Tuple[str, bool]]:
    results: List[Tuple[str, bool]] = []
    for p in paths:
        if not p:
            results.append((p, False))
            continue
        try:
            pathlib.Path(p).mkdir(parents=True, exist_ok=True)
            results.append((p, True))
        except Exception:
            results.append((p, False))
    return results


def check_hf_caches() -> Tuple[HFCaches, List[Tuple[str, bool]]]:
    env = check_env_vars()
    caches = HFCaches(
        hf_home=env.get("HF_HOME"),
        datasets_cache=env.get("HF_DATASETS_CACHE"),
        hub_cache=env.get("HF_HUB_CACHE"),
        transformers_cache=env.get("TRANSFORMERS_CACHE"),
    )
    ensured = ensure_dirs(
        [
            caches.hf_home or "",
            caches.datasets_cache or "",
            caches.hub_cache or "",
        ]
    )
    return caches, ensured


@dataclasses.dataclass
class ModelCheck:
    tokenizer_cached: bool
    model_cached: bool
    tokenizer_loadable: bool
    model_loadable: bool
    snapshot_dir: Optional[str]
    error: Optional[str]


def check_model_and_tokenizer(model_id: str = MODEL_ID) -> ModelCheck:
    tokenizer_cached = False
    model_cached = False
    tokenizer_loadable = False
    model_loadable = False
    snapshot_dir: Optional[str] = None
    error: Optional[str] = None

    try:
        from huggingface_hub import snapshot_download  # type: ignore
        # Tokenizer presence (local only)
        try:
            snapshot_download(
                repo_id=model_id,
                allow_patterns=[
                    "tokenizer*",
                    "vocab*",
                    "merges*",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "tokenizer.model",
                ],
                local_files_only=True,
            )
            tokenizer_cached = True
        except Exception:
            tokenizer_cached = False

        # Full snapshot presence (local only)
        try:
            snapshot_dir = snapshot_download(
                repo_id=model_id,
                local_files_only=True,
            )
            model_cached = True
        except Exception:
            model_cached = False

        # Loadability via transformers (local only)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
            try:
                _ = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                tokenizer_loadable = True
            except Exception:
                tokenizer_loadable = False

            try:
                _ = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
                model_loadable = True
            except Exception:
                model_loadable = False
        except Exception as exc:
            # transformers not available or other error
            error = f"transformers check failed: {exc}"
    except Exception as exc:
        error = f"huggingface_hub check failed: {exc}"

    return ModelCheck(
        tokenizer_cached=tokenizer_cached,
        model_cached=model_cached,
        tokenizer_loadable=tokenizer_loadable,
        model_loadable=model_loadable,
        snapshot_dir=snapshot_dir,
        error=error,
    )


@dataclasses.dataclass
class AccelerateCheck:
    config_exists: bool
    cli_available: bool


def check_accelerate() -> AccelerateCheck:
    cfg_exists = ACCELERATE_CFG.exists()
    code, _, _ = run_cmd(["bash", "-lc", "command -v accelerate >/dev/null 2>&1 && echo OK || true"])
    cli_available = (code == 0)
    return AccelerateCheck(config_exists=cfg_exists, cli_available=cli_available)


@dataclasses.dataclass
class WandbCheck:
    disabled: bool
    mode_offline: bool
    silent: bool


def check_wandb() -> WandbCheck:
    env = check_env_vars()
    disabled = str(env.get("WANDB_DISABLED", "")).lower() in {"1", "true", "yes"}
    mode_offline = str(env.get("WANDB_MODE", "")).lower() == "offline"
    silent = str(env.get("WANDB_SILENT", "")).lower() in {"1", "true", "yes"}
    return WandbCheck(disabled=disabled, mode_offline=mode_offline, silent=silent)


@dataclasses.dataclass
class GpuCheck:
    nvidia_smi_L: str
    nvidia_smi_query: str
    nvcc_version: str
    torch_cuda_available: Optional[bool]
    torch_num_devices: Optional[int]
    torch_device0_name: Optional[str]


def _detect_cuda_info() -> str:
    """Return a multiline string describing CUDA toolchain versions.

    Tries in order:
    - nvcc --version / -V
    - /usr/local/cuda/version.txt
    - nvidia-smi header line (CUDA Version: X.Y)
    - torch.version.cuda
    """
    parts: List[str] = []

    # nvcc --version
    for cmd in [
        "nvcc --version",
        "nvcc -V",
    ]:
        code, out, _ = run_cmd(["bash", "-lc", f"{cmd} 2>/dev/null || true"])
        if out:
            parts.append(f"{cmd}:\n{out}")
            break

    # /usr/local/cuda/version.txt
    try:
        p = pathlib.Path("/usr/local/cuda/version.txt")
        if p.exists():
            txt = p.read_text().strip()
            if txt:
                parts.append(f"/usr/local/cuda/version.txt: {txt}")
    except Exception:
        pass

    # nvidia-smi header
    _, smi_head, _ = run_cmd(["bash", "-lc", "nvidia-smi 2>/dev/null | head -n 1 || true"])
    if smi_head:
        parts.append(f"nvidia-smi header: {smi_head}")

    # torch.version.cuda
    try:
        import torch  # type: ignore

        if getattr(torch, "version", None) is not None:
            cuda_v = getattr(torch.version, "cuda", None)
            if cuda_v:
                parts.append(f"torch.version.cuda: {cuda_v}")
    except Exception:
        pass

    return "\n".join(parts).strip()


def check_gpu() -> GpuCheck:
    _, smi_L, _ = run_cmd(["bash", "-lc", "nvidia-smi -L 2>/dev/null || true"])
    _, smi_Q, _ = run_cmd([
        "bash",
        "-lc",
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true",
    ])
    nvcc_v = _detect_cuda_info()

    torch_available = None
    torch_devices = None
    torch_dev0 = None
    try:
        import torch  # type: ignore

        torch_available = bool(torch.cuda.is_available())
        torch_devices = int(torch.cuda.device_count())
        if torch_available and torch_devices and torch_devices > 0:
            torch_dev0 = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return GpuCheck(
        nvidia_smi_L=smi_L,
        nvidia_smi_query=smi_Q,
        nvcc_version=nvcc_v,
        torch_cuda_available=torch_available,
        torch_num_devices=torch_devices,
        torch_device0_name=torch_dev0,
    )


def write_hardware_md(gpu: GpuCheck) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    content = (
        "# Hardware Snapshot\n\n"
        f"- Timestamp (UTC): {ts}\n"
        "- nvidia-smi -L:\n\n"
        "```\n"
        f"{gpu.nvidia_smi_L}\n"
        "```\n\n"
        "- GPU name, driver, memory (from nvidia-smi):\n\n"
        "```\n"
        f"{gpu.nvidia_smi_query}\n"
        "```\n\n"
        "- nvcc --version (if available):\n\n"
        "```\n"
        f"{gpu.nvcc_version}\n"
        "```\n"
    )
    HARDWARE_MD.write_text(content)


def build_report(env: Dict[str, Optional[str]], caches: HFCaches, ensured: List[Tuple[str, bool]],
                 model: ModelCheck, acc: AccelerateCheck, wb: WandbCheck, gpu: GpuCheck) -> str:
    lines: List[str] = []
    lines.append("=== Self-check Report ===")
    lines.append("")
    lines.append("[Environment]")
    lines.append(f"CONDA_PREFIX: {env.get('CONDA_PREFIX')}")
    lines.append("")
    lines.append("[Hugging Face Cache]")
    lines.append(f"HF_HOME: {caches.hf_home}")
    lines.append(f"HF_DATASETS_CACHE: {caches.datasets_cache}")
    lines.append(f"HF_HUB_CACHE: {caches.hub_cache}")
    lines.append(f"TRANSFORMERS_CACHE: {caches.transformers_cache}")
    for path, ok in ensured:
        lines.append(f"ensure_dir {path!r}: {'OK' if ok else 'FAIL'}")
    lines.append("")
    lines.append("[Model/Tokenizer: Qwen/Qwen2.5-7B-Instruct]")
    lines.append(f"tokenizer_cached: {model.tokenizer_cached}")
    lines.append(f"model_cached: {model.model_cached}")
    lines.append(f"tokenizer_loadable: {model.tokenizer_loadable}")
    lines.append(f"model_loadable: {model.model_loadable}")
    lines.append(f"snapshot_dir: {model.snapshot_dir}")
    lines.append(f"error: {model.error}")
    lines.append("")
    lines.append("[Accelerate]")
    lines.append(f"config_exists: {acc.config_exists} path={ACCELERATE_CFG}")
    lines.append(f"cli_available: {acc.cli_available}")
    lines.append("")
    lines.append("[Weights & Biases]")
    lines.append(f"WANDB_DISABLED: {wb.disabled}")
    lines.append(f"WANDB_MODE_offline: {wb.mode_offline}")
    lines.append(f"WANDB_SILENT: {wb.silent}")
    lines.append("")
    lines.append("[GPU]")
    lines.append(f"nvidia-smi -L:\n{gpu.nvidia_smi_L}")
    lines.append(f"query (name,driver,mem):\n{gpu.nvidia_smi_query}")
    lines.append(f"nvcc: {gpu.nvcc_version}")
    lines.append(
        f"torch cuda_available={gpu.torch_cuda_available} num_devices={gpu.torch_num_devices} dev0={gpu.torch_device0_name}"
    )
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    write_doc = True
    if "--no-write" in argv:
        write_doc = False

    env = check_env_vars()
    caches, ensured = check_hf_caches()
    model = check_model_and_tokenizer(MODEL_ID)
    acc = check_accelerate()
    wb = check_wandb()
    gpu = check_gpu()

    if write_doc:
        try:
            write_hardware_md(gpu)
        except Exception as exc:
            print(f"Failed to write {HARDWARE_MD}: {exc}", file=sys.stderr)

    report = build_report(env, caches, ensured, model, acc, wb, gpu)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


