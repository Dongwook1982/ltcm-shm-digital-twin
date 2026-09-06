"""
LTCM -- Physics-Informed Neural Network Training Scripts
Signal-Driven Lifecycle Memory for SHM Digital Twins
Mechanical Systems and Signal Processing, 2025
Dong-Wook Kim, DL E&C Co., Ltd.

Three PINN degradation models (Section 5.1 of manuscript):
  1. Track settlement  -- Sato-Odaka model  (Case A: Railway viaduct)
  2. Fatigue crack     -- Paris law          (Case C: PSC bridge)
  3. Bearing degradation -- Power-law        (Case C: PSC bridge)

NOTE: Operational data is confidential. This script uses synthetic
data generators that replicate corpus statistics reported in the manuscript.
Replace data loaders with actual operational data for full reproduction.
Physics regularization and architecture exactly match manuscript (Sec 5.1).

Dependencies:
  torch >= 2.0, numpy >= 1.24, scikit-learn >= 1.3, scipy >= 1.11

Usage:
  python pinn_training.py --mechanism track_settlement --corpus_years 12
  python pinn_training.py --mechanism fatigue_crack    --corpus_years 32
  python pinn_training.py --mechanism bearing          --corpus_years 32

Corpus-size sensitivity (reproduces Fig. 5 right panel):
  for yr in 1 3 5 7 10 12; do
      python pinn_training.py --mechanism track_settlement --corpus_years $yr
  done
"""

import argparse, json, os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Configuration matches ltcm_config.yaml exactly
CONFIG = {
    "track_settlement": {
        "hidden_layers": 4, "neurons": 64,
        "governing_eq": "Sato-Odaka: d = alpha * N^beta",
        "alpha_range": (0.9, 1.5), "beta_range": (0.68, 0.75),
        "threshold_mm": 15.0,          # KR C-14020 absolute limit
        "description": "Track settlement RMSE converted to remaining service life (yr) via threshold-crossing time"
    },
    "fatigue_crack": {
        "hidden_layers": 5, "neurons": 64,
        "governing_eq": "Paris law: da/dN = C*(delta_K)^m",
        "C_range": (1e-10, 1e-9), "m_range": (2.5, 4.0),
        "critical_crack_mm": 25.0,     # Design limit
        "description": "Fatigue crack RMSE converted to remaining service life (yr) via threshold-crossing time"
    },
    "bearing": {
        "hidden_layers": 4, "neurons": 64,
        "governing_eq": "Power-law mass-loss: M_loss = k * t^n",
        "k_range": (0.03, 0.06), "n_range": (1.0, 1.4),
        "design_life_yr": 20.0,        # KDS 24 10 20
        "coastal_factor": 1.4,
        "description": "Bearing degradation RMSE converted to remaining service life (yr) via threshold-crossing time"
    },
    "training": {
        "optimizer": "Adam",
        "lr": 1e-3, "beta1": 0.9, "beta2": 0.999,
        "lambda_phys": 0.1,            # Physics-to-data loss weight
        "max_epochs": 5000,
        "patience": 200,               # Early stopping on validation loss
        "train_frac": 0.70,
        "val_frac":   0.15,
        "test_frac":  0.15,            # All splits by temporal order
        "mc_dropout_passes": 100,      # T for MC Dropout uncertainty
        "dropout_rate": 0.1,
    }
}


class PINNNetwork(nn.Module):
    """
    Physics-Informed Neural Network with MC Dropout (Sec 5.1).
    Architecture: n_hidden x n_neurons, tanh activation.
    Uncertainty: Monte Carlo Dropout (T=100 forward passes, 95% CI).
    """
    def __init__(self, n_input, n_output, n_hidden, n_neurons, dropout_rate=0.1):
        super().__init__()
        layers = [nn.Linear(n_input, n_neurons), nn.Tanh(), nn.Dropout(dropout_rate)]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh(), nn.Dropout(dropout_rate)]
        layers.append(nn.Linear(n_neurons, n_output))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def mc_predict(self, x, T=100):
        """Monte Carlo Dropout for 95% confidence interval (Sec 5.1)."""
        self.train()
        preds = torch.stack([self(x) for _ in range(T)], dim=0)
        self.eval()
        mean = preds.mean(0)
        std  = preds.std(0)
        return mean, mean - 1.96*std, mean + 1.96*std


def physics_residual_track_settlement(model, N_norm, alpha, beta):
    """Sato-Odaka residual: d_pred - alpha*(N_actual^beta) = 0."""
    N_norm.requires_grad_(True)
    d_pred = model(N_norm.unsqueeze(-1))
    d_target = torch.tensor(alpha, dtype=torch.float32) * (N_norm ** beta)
    return nn.MSELoss()(d_pred.squeeze(), d_target)


def physics_residual_fatigue(model, N_norm, C, m, delta_K=10.0):
    """
    Paris law residual: da/dN - C*(delta_K)^m = 0.
    Uses autograd to compute da/dN from predicted crack length a(N).
    delta_K = stress intensity factor range (MPa*sqrt(m)), typical bridge value.
    """
    N_norm = N_norm.clone().detach().requires_grad_(True)
    a_pred = model(N_norm.unsqueeze(-1))
    da_dN  = torch.autograd.grad(a_pred.sum(), N_norm, create_graph=True)[0]
    da_dN_physics = C * (delta_K ** m)
    return nn.MSELoss()(da_dN, torch.full_like(da_dN, da_dN_physics))


def physics_residual_bearing(model, t_norm, k, n_exp):
    """Power-law residual: M_pred - k*t^n = 0."""
    t_norm.requires_grad_(True)
    M_pred  = model(t_norm.unsqueeze(-1))
    M_target = torch.tensor(k, dtype=torch.float32) * (t_norm ** n_exp)
    return nn.MSELoss()(M_pred.squeeze(), M_target)


def generate_track_settlement(n_years=12, n_spans=18, surveys_per_year=2):
    """
    Synthetic track settlement data replicating Case A statistics.
    Temporal 70/15/15 split (no future leakage).
    Note: RMSE output converted to remaining-service-life years via
    threshold-crossing time (KR C-14020: 15 mm absolute limit).
    """
    np.random.seed(SEED)
    alphas = np.random.uniform(*CONFIG["track_settlement"]["alpha_range"], n_spans)
    betas  = np.random.uniform(*CONFIG["track_settlement"]["beta_range"],  n_spans)
    threshold = CONFIG["track_settlement"]["threshold_mm"]
    records = []
    for span in range(n_spans):
        for yr in np.linspace(0.5, n_years, n_years * surveys_per_year):
            N_actual = yr * 1e6                   # cycles per year
            d_mm = alphas[span] * (N_actual ** betas[span]) / 1e6
            d_obs = d_mm + np.random.normal(0, 0.3)
            # Remaining service life = time to threshold crossing
            if d_mm < threshold:
                # Solve: threshold = alpha * N^beta for N, convert back to years
                N_thresh = (threshold / alphas[span]) ** (1/betas[span])
                rsl_yr = max(0.0, (N_thresh - N_actual) / 1e6)
            else:
                rsl_yr = 0.0
            records.append({"year": yr, "N_norm": yr/n_years,
                            "d_mm_obs": d_obs, "rsl_yr": rsl_yr,
                            "alpha": alphas[span], "beta": betas[span]})
    sorted_r = sorted(records, key=lambda x: x["year"])
    n = len(sorted_r); n_tr = int(n*0.70); n_v = int(n*0.15)
    return {"train": sorted_r[:n_tr], "val": sorted_r[n_tr:n_tr+n_v],
            "test":  sorted_r[n_tr+n_v:],
            "metadata": {"n_years": n_years, "n_spans": n_spans,
                         "model": "Sato-Odaka", "target": "RSL (yr)"}}


def generate_fatigue_crack(n_events=60, years_span=32):
    """
    Synthetic fatigue crack data replicating Case C Paris-law validation.
    RMSE converted to remaining service life (yr) via threshold-crossing time
    (critical crack length: 25 mm).
    """
    np.random.seed(SEED + 2)
    C  = np.random.uniform(*CONFIG["fatigue_crack"]["C_range"])
    m  = np.random.uniform(*CONFIG["fatigue_crack"]["m_range"])
    a0 = 0.5   # initial crack length (mm)
    delta_K = 10.0
    critical = CONFIG["fatigue_crack"]["critical_crack_mm"]
    t_arr = np.sort(np.random.uniform(0, years_span, n_events))
    records = []
    for t in t_arr:
        N_cycles = t * 1e6
        a = a0 + C * (N_cycles**m) / 1e18 * 1e4   # scaled for numerics
        a_obs = a + np.random.normal(0, 0.05)
        # RSL: remaining cycles / 1e6 yr until critical crack
        # Solve: critical = a0 + C*(N_crit^m)/scale for N_crit
        N_crit_scaled = ((critical - a0) / (C * 1e4 / 1e18)) ** (1/m)
        rsl_yr = max(0.0, (N_crit_scaled - N_cycles) / 1e6)
        records.append({"year": t, "N_norm": t/years_span,
                        "a_mm_obs": float(a_obs), "rsl_yr": float(rsl_yr),
                        "C": C, "m": m})
    sorted_r = sorted(records, key=lambda x: x["year"])
    n = len(sorted_r); n_tr = int(n*0.70); n_v = int(n*0.15)
    return {"train": sorted_r[:n_tr], "val": sorted_r[n_tr:n_tr+n_v],
            "test":  sorted_r[n_tr+n_v:],
            "metadata": {"n_events": n_events, "years": years_span,
                         "model": "Paris law", "target": "RSL (yr)"}}


def generate_bearing(n_events=14, years_span=32):
    """
    Synthetic bearing degradation data (Case C, 14 replacement events).
    RSL: remaining years until mass-loss replacement criterion.
    """
    np.random.seed(SEED + 1)
    k = np.random.uniform(*CONFIG["bearing"]["k_range"])
    n = np.random.uniform(*CONFIG["bearing"]["n_range"])
    design_life = CONFIG["bearing"]["design_life_yr"] / CONFIG["bearing"]["coastal_factor"]
    t_arr = np.sort(np.random.uniform(0, years_span, n_events))
    records = []
    for t in t_arr:
        M_loss = k * (t ** n)
        M_obs  = M_loss + np.random.normal(0, 0.05)
        # RSL: replacement when M_loss = k * t_repl^n at design life
        M_criterion = k * (design_life ** n)
        if M_loss < M_criterion:
            t_repl = (M_criterion / k) ** (1/n)
            rsl_yr = max(0.0, t_repl - t)
        else:
            rsl_yr = 0.0
        records.append({"year": float(t), "t_norm": float(t/years_span),
                        "M_loss_kg": float(M_obs), "rsl_yr": float(rsl_yr),
                        "k": k, "n_exp": n})
    sorted_r = sorted(records, key=lambda x: x["year"])
    n_rec = len(sorted_r); n_tr = int(n_rec*0.70); n_v = int(n_rec*0.15)
    return {"train": sorted_r[:n_tr], "val": sorted_r[n_tr:n_tr+n_v],
            "test":  sorted_r[n_tr+n_v:],
            "metadata": {"n_events": n_events, "years": years_span,
                         "model": "Power-law M_loss=k*t^n", "target": "RSL (yr)"}}


def train_pinn(mechanism, corpus_years=12, verbose=True):
    cfg_m = CONFIG[mechanism]; cfg_t = CONFIG["training"]
    print(f"\nPINN Training: {mechanism.upper()} | {cfg_t['max_epochs']} epochs max")
    print(f"  Governing equation: {cfg_m['governing_eq']}")
    print(f"  Target: {cfg_m['description']}")
    print(f"  Corpus: {corpus_years} years | Lambda_phys: {cfg_t['lambda_phys']}")

    # Data
    if mechanism == "track_settlement":
        data = generate_track_settlement(n_years=corpus_years)
        X_key, y_key = "N_norm", "rsl_yr"
    elif mechanism == "fatigue_crack":
        data = generate_fatigue_crack(years_span=corpus_years)
        X_key, y_key = "N_norm", "rsl_yr"
    else:
        data = generate_bearing(n_events=min(14, max(5, corpus_years//2)),
                                years_span=corpus_years)
        X_key, y_key = "t_norm", "rsl_yr"

    def to_tensor(recs):
        X = torch.tensor([r[X_key] for r in recs], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor([r[y_key] for r in recs], dtype=torch.float32).unsqueeze(-1)
        return X, y

    X_tr, y_tr = to_tensor(data["train"])
    X_v,  y_v  = to_tensor(data["val"])
    X_te, y_te = to_tensor(data["test"])

    model = PINNNetwork(1, 1, cfg_m["hidden_layers"], cfg_m["neurons"],
                        cfg_t["dropout_rate"])
    opt   = optim.Adam(model.parameters(), lr=cfg_t["lr"],
                       betas=(cfg_t["beta1"], cfg_t["beta2"]))
    mse   = nn.MSELoss()
    lp    = cfg_t["lambda_phys"]
    best_val, patience_cnt = float("inf"), 0
    best_state = None

    for epoch in range(cfg_t["max_epochs"]):
        model.train(); opt.zero_grad()
        y_hat = model(X_tr)
        data_loss = mse(y_hat, y_tr)

        # Physics residual
        if mechanism == "track_settlement":
            am = (cfg_m["alpha_range"][0]+cfg_m["alpha_range"][1])/2
            bm = (cfg_m["beta_range"][0] +cfg_m["beta_range"][1])/2
            phys = physics_residual_track_settlement(model, X_tr.squeeze(), am, bm)
        elif mechanism == "fatigue_crack":
            Cm = (cfg_m["C_range"][0]+cfg_m["C_range"][1])/2
            mm = (cfg_m["m_range"][0] +cfg_m["m_range"][1])/2
            phys = physics_residual_fatigue(model, X_tr.squeeze(), Cm, mm)
        else:
            km = (cfg_m["k_range"][0]+cfg_m["k_range"][1])/2
            nm = (cfg_m["n_range"][0]+cfg_m["n_range"][1])/2
            phys = physics_residual_bearing(model, X_tr.squeeze(), km, nm)

        loss = data_loss + lp * phys
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_rmse = torch.sqrt(mse(model(X_v), y_v)).item()

        if val_rmse < best_val:
            best_val = val_rmse; patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        if patience_cnt >= cfg_t["patience"]:
            if verbose: print(f"  Early stopping at epoch {epoch+1}")
            break
        if verbose and (epoch+1) % 1000 == 0:
            print(f"  Epoch {epoch+1:4d} | data={data_loss.item():.4f} "
                  f"val_rmse={val_rmse:.4f}")

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        test_rmse = torch.sqrt(mse(model(X_te), y_te)).item()
    _, ci_lo, ci_hi = model.mc_predict(X_te, cfg_t["mc_dropout_passes"])

    print(f"  Test RMSE (RSL yr): {test_rmse:.4f}")
    print(f"  Mean 95% CI width:  {(ci_hi-ci_lo).mean().item():.4f}")
    print(f"  Train/val/test:     {len(data['train'])}/{len(data['val'])}/{len(data['test'])}")

    return model, {"mechanism": mechanism, "corpus_years": corpus_years,
                   "test_rmse_yr": test_rmse, "n_train": len(data["train"]),
                   "target_unit": "remaining service life (yr)"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTCM PINN Training (Sec 5.1)")
    parser.add_argument("--mechanism",
                        choices=["track_settlement","fatigue_crack","bearing"],
                        default="track_settlement")
    parser.add_argument("--corpus_years", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, results = train_pinn(args.mechanism, args.corpus_years)
    out_path = os.path.join(args.output_dir,
                            f"pinn_{args.mechanism}_{args.corpus_years}yr_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Corpus sensitivity (Fig. 5 right panel: settlement only for speed)
    if args.mechanism == "track_settlement":
        print("\nCorpus-size sensitivity (Fig. 5 right panel):")
        for yr in [1, 3, 5, 7, 10, 12]:
            if yr > args.corpus_years: break
            _, res = train_pinn(args.mechanism, yr, verbose=False)
            print(f"  {yr:2d} yr corpus -> RMSE = {res['test_rmse_yr']:.4f} yr")
