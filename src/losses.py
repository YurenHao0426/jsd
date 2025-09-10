# -*- coding: utf-8 -*-
"""
Losses for:
- EM (entropy minimization)
- Group-EM (entropy-difference between female/male token groups)
- JSD counterfactual invariance (x vs swap(x)) with optional Top-K
Guards:
- Mass parity: (piF - piM)^2
- Stability:   KL( p_theta || p_base )
Gating:
- Top-K trigger on {F ∪ M} at each step (boundary-safe happens at text level during data build/eval)
Note:
- All losses are averaged over steps where gate==1 AND (optionally) generation mask==1.
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

def map_words_to_token_ids(tok, words: List[str]) -> List[int]:
    ids = set()
    for w in words:
        for form in (w, " " + w):
            enc = tok(form, add_special_tokens=False, return_tensors=None)
            toks = enc["input_ids"]
            if len(toks) == 1:
                ids.add(int(toks[0]))
            elif len(toks) > 1:
                ids.add(int(toks[0]))  # first-piece fallback
    return sorted(ids)

def probs_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        # avoid div-by-zero; at T=0 use softmax on raw logits (equivalent to no scaling)
        return F.softmax(logits, dim=-1)
    return F.softmax(logits / temperature, dim=-1)

def topk_gate(logits: torch.Tensor, fem_ids: List[int], male_ids: List[int], k: int = 20) -> torch.Tensor:
    """
    logits: [B,T,V]
    Return gate mask [B,T] == 1 if top-k at step contains any F∪M id.
    """
    B,T,V = logits.shape
    topk = torch.topk(logits, k=min(k, V), dim=-1).indices  # [B,T,k]
    ids = torch.tensor(list(set(fem_ids) | set(male_ids)), device=logits.device, dtype=torch.long)
    if ids.numel() == 0:
        return torch.zeros(B,T, dtype=torch.float32, device=logits.device)
    # Compare with broadcasting
    match = (topk.unsqueeze(-1) == ids.view(1,1,1,-1)).any(dim=-1)  # [B,T,k] -> [B,T]
    return match.float()

def group_masses(probs: torch.Tensor, fem_ids: List[int], male_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    probs: [B,T,V]
    Returns piF, piM of shape [B,T]
    """
    if len(fem_ids) == 0 and len(male_ids) == 0:
        return torch.zeros_like(probs[...,0]), torch.zeros_like(probs[...,0])
    idxF = torch.tensor(fem_ids, device=probs.device, dtype=torch.long) if len(fem_ids)>0 else None
    idxM = torch.tensor(male_ids, device=probs.device, dtype=torch.long) if len(male_ids)>0 else None
    piF = probs[..., idxF].sum(dim=-1) if idxF is not None else torch.zeros_like(probs[...,0])
    piM = probs[..., idxM].sum(dim=-1) if idxM is not None else torch.zeros_like(probs[...,0])
    return piF, piM

def normalized_entropy(sub_probs: torch.Tensor) -> torch.Tensor:
    """
    sub_probs: [*, K]
    Return normalized entropy in [0,1]: H(p)/log(K)
    """
    eps = 1e-12
    K = sub_probs.size(-1)
    H = -(sub_probs.clamp_min(eps) * sub_probs.clamp_min(eps).log()).sum(dim=-1)
    denom = torch.log(torch.tensor(float(K), device=sub_probs.device))
    return H / (denom + eps)

def group_entropies(probs: torch.Tensor, fem_ids: List[int], male_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    probs: [B,T,V] -> qF [B,T,|F|], qM [B,T,|M|] -> HF, HM in [0,1]
    """
    eps = 1e-12
    idxF = torch.tensor(fem_ids, device=probs.device, dtype=torch.long) if len(fem_ids)>0 else None
    idxM = torch.tensor(male_ids, device=probs.device, dtype=torch.long) if len(male_ids)>0 else None

    if idxF is None:
        HF = torch.zeros(probs.shape[:2], device=probs.device)
    else:
        pF = probs[..., idxF]                          # [B,T,|F|]
        piF = pF.sum(dim=-1, keepdim=True) + eps
        qF = pF / piF
        HF = normalized_entropy(qF)

    if idxM is None:
        HM = torch.zeros(probs.shape[:2], device=probs.device)
    else:
        pM = probs[..., idxM]
        piM = pM.sum(dim=-1, keepdim=True) + eps
        qM = pM / piM
        HM = normalized_entropy(qM)

    return HF, HM

def reduce_steps(x: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:
    """
    x: [B,T], step_mask: [B,T] in {0,1}
    Return mean over steps where mask==1 (avoid div by 0).
    """
    w = step_mask
    s = (x * w).sum()
    d = w.sum().clamp_min(1.0)
    return s / d

# ---------------- EM ----------------
def loss_em(logits: torch.Tensor, gen_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """
    Entropy minimization over generation steps (no gating).
    logits: [B,T,V]; gen_mask: [B,T] 1 for generation steps (non-prompt)
    """
    probs = probs_from_logits(logits)                 # [B,T,V]
    eps = 1e-12
    Ht = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)  # [B,T]
    L = reduce_steps(Ht, gen_mask)
    return L, {"H_mean": float(reduce_steps(Ht, gen_mask).item())}

# ------------- Group-EM -------------
def loss_group_em(
    logits: torch.Tensor,
    gen_mask: torch.Tensor,
    fem_ids: List[int],
    male_ids: List[int],
    gate_mask: Optional[torch.Tensor] = None,
    lambda_mass: float = 0.0,
    beta_kl: float = 0.0,
    ref_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Group-EM loss with optional guards.
    - core: (H_F - H_M)^2
    - mass guard: (piF - piM)^2
    - stability: KL( p || pref )
    """
    probs = probs_from_logits(logits)                 # [B,T,V]
    HF, HM = group_entropies(probs, fem_ids, male_ids)  # [B,T], [B,T]
    core = (HF - HM) ** 2                             # [B,T]

    piF, piM = group_masses(probs, fem_ids, male_ids)
    Lmass = (piF - piM) ** 2                          # [B,T]

    if gate_mask is None:
        step_mask = gen_mask
    else:
        step_mask = (gen_mask * gate_mask).float()

    L_core = reduce_steps(core, step_mask)
    L_mass = reduce_steps(Lmass, step_mask)

    L_kl = torch.tensor(0.0, device=logits.device)
    if beta_kl > 0.0 and ref_probs is not None:
        eps = 1e-12
        p = probs.clamp_min(eps)
        q = ref_probs.clamp_min(eps)
        KL = (p * (p.log() - q.log())).sum(dim=-1)    # [B,T]
        L_kl = reduce_steps(KL, step_mask)

    loss = L_core + lambda_mass * L_mass + beta_kl * L_kl
    extras = {
        "L_core": float(L_core.item()),
        "L_mass": float(L_mass.item()),
        "L_kl": float(L_kl.item()) if isinstance(L_kl, torch.Tensor) else float(L_kl),
        "piF_mean": float(reduce_steps(piF, step_mask).item()),
        "piM_mean": float(reduce_steps(piM, step_mask).item()),
        "HF_mean": float(reduce_steps(HF, step_mask).item()),
        "HM_mean": float(reduce_steps(HM, step_mask).item()),
    }
    return loss, extras

# --------------- JSD ---------------
def _jsd_full(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * (p * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1) + \
           0.5 * (q * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)

def _jsd_topk(p: torch.Tensor, q: torch.Tensor, K: int) -> torch.Tensor:
    V = p.size(-1)
    K = min(K, V)
    idx_p = torch.topk(p, k=K, dim=-1).indices
    idx_q = torch.topk(q, k=K, dim=-1).indices
    idx = torch.cat([idx_p, idx_q], dim=-1).unique(dim=-1)  # union
    pK = p.gather(-1, idx); qK = q.gather(-1, idx)
    mK = 0.5 * (pK + qK)
    eps = 1e-12
    return 0.5 * (pK * (pK.clamp_min(eps).log() - mK.clamp_min(eps).log())).sum(dim=-1) + \
           0.5 * (qK * (qK.clamp_min(eps).log() - mK.clamp_min(eps).log())).sum(dim=-1)

def loss_jsd(
    logits_f: torch.Tensor,   # [B,T,V]
    logits_c: torch.Tensor,   # [B,T,V]
    gen_mask: torch.Tensor,   # [B,T]
    fem_ids: List[int],
    male_ids: List[int],
    gate_mask_f: Optional[torch.Tensor] = None,
    gate_mask_c: Optional[torch.Tensor] = None,
    lambda_mass: float = 0.0,
    beta_kl: float = 0.0,
    ref_probs_f: Optional[torch.Tensor] = None,
    topk_jsd: int = 0
) -> Tuple[torch.Tensor, Dict]:
    """
    JSD(p||q) averaged over steps with gating (factual and counterfactual separately gated).
    Also includes mass parity on both branches and optional stability to base on factual branch.
    """
    p = probs_from_logits(logits_f)                   # [B,T,V]
    q = probs_from_logits(logits_c)                   # [B,T,V]

    if topk_jsd and topk_jsd > 0:
        J = _jsd_topk(p, q, K=topk_jsd)              # [B,T]
    else:
        J = _jsd_full(p, q)                          # [B,T]

    # step mask: require gate on factual (and optionally also on counterfactual)
    if gate_mask_f is None:
        step_mask = gen_mask
    else:
        step_mask = (gen_mask * gate_mask_f).float()

    L_jsd = reduce_steps(J, step_mask)

    # mass parity on each branch
    piF_f, piM_f = group_masses(p, fem_ids, male_ids)
    piF_c, piM_c = group_masses(q, fem_ids, male_ids)
    L_mass = reduce_steps((piF_f - piM_f)**2, step_mask) + reduce_steps((piF_c - piM_c)**2, step_mask)

    # stability to base (factual branch)
    L_kl = torch.tensor(0.0, device=logits_f.device)
    if beta_kl > 0.0 and ref_probs_f is not None:
        eps = 1e-12
        p0 = ref_probs_f.clamp_min(eps)
        L_kl = reduce_steps((p.clamp_min(eps) * (p.clamp_min(eps).log() - p0.log())).sum(dim=-1), step_mask)

    loss = L_jsd + lambda_mass * L_mass + beta_kl * L_kl
    extras = {
        "L_jsd": float(L_jsd.item()),
        "L_mass": float(L_mass.item()),
        "L_kl": float(L_kl.item()) if isinstance(L_kl, torch.Tensor) else float(L_kl),
        "piF_f": float(reduce_steps(piF_f, step_mask).item()),
        "piM_f": float(reduce_steps(piM_f, step_mask).item()),
        "piF_c": float(reduce_steps(piF_c, step_mask).item()),
        "piM_c": float(reduce_steps(piM_c, step_mask).item()),
    }
    return loss, extras
