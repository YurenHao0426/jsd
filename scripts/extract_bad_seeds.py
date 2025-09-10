#!/usr/bin/env python3
import json, sys, pathlib

IN = pathlib.Path("third_party/bad_seeds/gathered_seeds.json")
OUT = pathlib.Path("assets/groups")
OUT.mkdir(parents=True, exist_ok=True)

def load_all():
    try:
        return json.loads(IN.read_text())
    except FileNotFoundError:
        sys.stderr.write(f"[ERR] Missing file: {IN}\n"); sys.exit(1)

def pick(data, id_exact=None, contains=None):
    """Return .Seeds list for a given entry by exact ID or substring match."""
    def _sid(x):
        return x.get("Seeds ID") or x.get("Seeds_ID") or x.get("SeedsID")
    for obj in data:
        sid = _sid(obj)
        if not sid: continue
        if id_exact is not None and sid == id_exact:
            return obj.get("Seeds")
    if contains:
        for obj in data:
            sid = _sid(obj) or ""
            if all(sub.lower() in sid.lower() for sub in contains):
                return obj.get("Seeds")
    return None

def dump(words, path):
    toks = sorted({(w or "").strip() for w in words if isinstance(w, str) and w.strip()})
    path.write_text("\n".join(toks) + "\n")
    return len(toks)

def main():
    data = load_all()

    # Canonical WEAT name sets (Caliskan et al. 2017)
    male = pick(data, id_exact="male_names_1-Caliskan_et_al_2017") or pick(data, contains=["male","name"])
    female = pick(data, id_exact="female_names_1-Caliskan_et_al_2017") or pick(data, contains=["female","name"])

    if not male or not female:
        # Help debug if schema changed
        sys.stderr.write("[ERR] Could not locate WEAT male/female name sets. Available IDs:\n")
        for obj in data:
            sid = obj.get("Seeds ID") or obj.get("Seeds_ID") or obj.get("SeedsID")
            if sid: sys.stderr.write("  - " + sid + "\n")
        sys.exit(2)

    n_m = dump(male, OUT / "weat_male_names.txt")
    n_f = dump(female, OUT / "weat_female_names.txt")

    # Optional: career/family word sets (also from Caliskan et al. 2017)
    career = pick(data, id_exact="career_words_1-Caliskan_et_al_2017") or pick(data, contains=["career"])
    family = pick(data, id_exact="family_words_1-Caliskan_et_al_2017") or pick(data, contains=["family"])
    n_c = n_fam = 0
    if career: n_c = dump(career, OUT / "weat_career_words.txt")
    if family: n_fam = dump(family, OUT / "weat_family_words.txt")

    print(f"Exported: male_names={n_m}, female_names={n_f}, career={n_c}, family={n_fam}")
if __name__ == "__main__":
    main()
