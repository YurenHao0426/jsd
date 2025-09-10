#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate training sets for:
- EM / Group-EM: 1, 5, 20 prompts (English only)
- JSD: 100 factual/counterfactual pairs (x, x_swap) with gender swap

Data sources:
  - assets/triggers/occupations_en.txt
  - assets/groups/en_definitional_pairs.json  (female<->male definitional pairs; supports multiple schemas)
  - assets/groups/en_female.txt / en_male.txt (for pronouns; we mainly use 'she'/'he')

Avoids overlap with eval CTF examples in data/bias/ctf/ctf_en.jsonl.

Outputs:
  data/train/em_group/train_en_size{1,5,20}.jsonl
  data/train/jsd/train_pairs_en_size100.jsonl
"""
import json, random, pathlib, re, sys
from typing import List, Tuple, Dict, Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
OUT_EM = DATA / "train" / "em_group"
OUT_JSD = DATA / "train" / "jsd"
EVAL_CTF = DATA / "bias" / "ctf" / "ctf_en.jsonl"

random.seed(2025)

def read_lines(p: pathlib.Path) -> List[str]:
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def read_jsonl(p: pathlib.Path) -> List[dict]:
    if not p.exists(): return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def _normalize_pair(a: Any, b: Any) -> Tuple[str,str]:
    return (str(a).strip().lower(), str(b).strip().lower())

def load_pairs_json_any(p: pathlib.Path) -> List[Tuple[str,str]]:
    """
    Robustly load definitional gender pairs from various schemas seen in the wild:
    - [ ["woman","man"], ["girl","boy"], ... ]
    - { "definitional": [ ["woman","man"], ... ] }
    - [ {"f":"woman","m":"man"}, ... ]  or keys {"female":...,"male":...}
    - { "pairs":[... same as above ...] }
    Returns a list of (female, male) lower-cased tuples.
    """
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    pairs: List[Tuple[str,str]] = []

    def add_from_list(lst: List[Any]):
        for item in lst:
            if isinstance(item, list) and len(item) == 2:
                a, b = item
                pairs.append(_normalize_pair(a, b))
            elif isinstance(item, dict):
                # common key patterns
                if "f" in item and "m" in item:
                    pairs.append(_normalize_pair(item["f"], item["m"]))
                elif "female" in item and "male" in item:
                    pairs.append(_normalize_pair(item["female"], item["male"]))
                elif "a" in item and "b" in item:
                    pairs.append(_normalize_pair(item["a"], item["b"]))

    if isinstance(data, list):
        add_from_list(data)
    elif isinstance(data, dict):
        for k in ("definitional", "definitional_pairs", "pairs"):
            if isinstance(data.get(k), list):
                add_from_list(data[k])

    # de-dup & sanity
    out = []
    seen = set()
    for a,b in pairs:
        if not a or not b or a == b: 
            continue
        key = (a,b)
        if key not in seen:
            seen.add(key); out.append(key)
    return out

def minimal_swap(text: str, pairs: List[Tuple[str,str]]) -> str:
    """
    Swap gendered tokens with word-boundary regex, preferring longer strings first.
    """
    if not pairs:
        return text  # no-op; caller will fallback to template-based swap
    mapping: Dict[str,str] = {}
    for a,b in pairs:
        mapping[a] = b
        mapping[b] = a
    # sort keys by length desc to avoid partial overshadow
    keys = sorted(mapping.keys(), key=len, reverse=True)
    out = text
    for k in keys:
        # boundary-safe, case-insensitive; replace with mapping[k] in lowercase
        pat = re.compile(rf"(?<!\w){re.escape(k)}(?!\w)", flags=re.IGNORECASE)
        out = pat.sub(mapping[k], out)
    return out

def sample_templates(occupations: List[str], n: int) -> List[Tuple[str,str,str]]:
    """
    Return list of tuples: (occupation, she_template, he_template)
    """
    SHE_FORMS = [
        "The {occ} said that she would review the design.",
        "The {occ} explained that she needed more time.",
        "The {occ} promised that she will lead the next meeting.",
        "The {occ} reported that she completed the task.",
        "The {occ} noted that she prefers the alternative plan.",
    ]
    HE_FORMS = [
        t.replace(" she ", " he ").replace("She ", "He ").replace(" she", " he")
        for t in SHE_FORMS
    ]
    random.shuffle(occupations)
    out = []
    i = 0
    while len(out) < n and i < 10*n:
        occ = occupations[i % len(occupations)]
        idx = random.randrange(len(SHE_FORMS))
        s_she = SHE_FORMS[idx].format(occ=occ)
        s_he  = HE_FORMS[idx].format(occ=occ)
        out.append((occ, s_she, s_he))
        i += 1
    return out[:n]

def main():
    OUT_EM.mkdir(parents=True, exist_ok=True)
    OUT_JSD.mkdir(parents=True, exist_ok=True)

    # sources
    occs = read_lines(ASSETS / "triggers" / "occupations_en.txt")
    pairs_json = ASSETS / "groups" / "en_definitional_pairs.json"
    pairs = load_pairs_json_any(pairs_json)
    eval_ctf = read_jsonl(EVAL_CTF)
    eval_x = set([r.get("x","").strip() for r in eval_ctf] + [r.get("x_swap","").strip() for r in eval_ctf])

    # ---- EM / Group-EM prompts (she-variant prompts; labels不需要)
    for size in [1,5,20]:
        triples = sample_templates(occs, size*3)  # oversample for filtering
        rows = []
        for occ, s_she, s_he in triples:
            if s_she in eval_x or s_he in eval_x:
                continue
            rows.append({"id": f"em_{len(rows):06d}", "lang":"en", "occupation": occ, "prompt": s_she})
            if len(rows) >= size: break
        outp = OUT_EM / f"train_en_size{size}.jsonl"
        outp.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
        print("Wrote", outp, "N=", len(rows))

    # ---- JSD pairs (x, x_swap)
    size_jsd = 100
    triples = sample_templates(occs, size_jsd*4)  # oversample more to be safe
    pairs_out = []
    for occ, s_she, s_he in triples:
        x = s_she
        if x in eval_x: 
            continue

        # try definitional-pair swap first
        x_swap = minimal_swap(x, pairs)
        # fallback: if no change, use our explicit he-template
        if x.strip().lower() == x_swap.strip().lower():
            x_swap = s_he

        if x_swap in eval_x: 
            continue
        if x.strip().lower() == x_swap.strip().lower():  # still identical? extremely unlikely
            continue

        pairs_out.append({
            "id": f"jsd_{len(pairs_out):06d}",
            "lang":"en",
            "occupation": occ,
            "prompt": x,
            "prompt_swap": x_swap
        })
        if len(pairs_out) >= size_jsd: break

    outp2 = OUT_JSD / "train_pairs_en_size100.jsonl"
    outp2.write_text("\n".join(json.dumps(r) for r in pairs_out) + ("\n" if pairs_out else ""), encoding="utf-8")
    print("Wrote", outp2, "N=", len(pairs_out))

    # quick diagnostics
    if len(pairs_out) == 0:
        print("[WARN] JSD pairs = 0. Diagnostics:")
        print("  - occupations:", len(occs))
        print("  - definitional pairs loaded:", len(pairs))
        print("  - eval_x size:", len(eval_x))
        print("  - Check assets/groups/en_definitional_pairs.json schema.")
        sys.exit(2)

if __name__ == "__main__":
    main()
