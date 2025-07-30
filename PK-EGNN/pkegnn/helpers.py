# ================================================================
# 7)  SE(3)‑equivariance & permutation‑invariance checker
# ================================================================
import torch, math, itertools, random
from collections import defaultdict
torch.set_printoptions(precision=3, sci_mode=True)

# ---------- helpers ----------------------------------------------------------
def _random_rotation(device):
    """Draw a random 3×3 rotation matrix from a unit quaternion."""
    q = torch.randn(4, device=device); q /= q.norm()
    w, x, y, z = q
    return torch.tensor([[1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                         [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
                         [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]],
                        device=device)

def _prep_batch(batch, cfg):
    """Flatten (B,S,K,…) → (R,K,…) and drop padded rows."""
    z, x, _, m, *_ = batch
    mask  = m.view(-1)                             # (R,)
    z = z.view(-1, z.size(2))[mask].to(cfg.device)           # (R,K)
    x = x.view(-1, x.size(2), 3)[mask].to(cfg.device)        # (R,K,3)
    return z, x                                     # R = Σ valid residues

# ---------- core checks ------------------------------------------------------
@torch.no_grad()
def run_invariance_suite(model, loader, cfg,
                         max_batches=4, rot_trials=3,
                         atol=5e-4, rtol=5e-4, verbose=True):
    """
    For `max_batches` mini‑batches:
      • SE(3) equivariance   (rot + trans)
      • neighbour perm‑inv   (perm K)
      • residue  perm‑eqv    (perm R)
    Returns dict with max abs‑errors.
    """
    stats = defaultdict(float)
    model.eval()

    for b_id, batch in enumerate(loader):
        if b_id >= max_batches: break
        z, x = _prep_batch(batch, cfg)             # (R,K), (R,K,3)

        # --- baseline prediction -------------------------------------------------
        base = model(z, x).flatten()               # (R,)

        # -- 1) SE(3) equivariance -----------------------------------------------
        for t in range(rot_trials):
            R = _random_rotation(x.device)
            tvec = torch.randn(1,1,3, device=x.device)
            x_rt = (x @ R.T) + tvec
            p_rt = model(z, x_rt).flatten()
            err  = (base - p_rt).abs().max().item()
            stats['eqv'] = max(stats['eqv'], err)
            if verbose:
                print(f"[batch {b_id}  rot {t}]  max|Δ|={err:.3e}")

        # -- 2) neighbour‑perm invariance ----------------------------------------
        K = z.size(1)
        permK = torch.randperm(K, device=x.device)
        pK = model(z[:, permK], x[:, permK]).flatten()
        errK = (base - pK).abs().max().item()
        stats['permK'] = max(stats['permK'], errK)
        if verbose:
            print(f"[batch {b_id}  perm‑K ]  max|Δ|={errK:.3e}")

        # -- 3) residue‑perm equivariance ----------------------------------------
        Rn = z.size(0)
        permR = torch.randperm(Rn, device=x.device)
        zR, xR  = z[permR], x[permR]
        pR = model(zR, xR).flatten()
        # undo permutation on prediction
        pR = pR[permR.argsort()]
        errR = (base - pR).abs().max().item()
        stats['permR'] = max(stats['permR'], errR)
        if verbose:
            print(f"[batch {b_id}  perm‑R ]  max|Δ|={errR:.3e}")
            print("-"*55)

    return stats
