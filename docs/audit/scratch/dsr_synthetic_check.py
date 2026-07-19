"""Scratch: verify calculate_dsr against canonical Bailey & Lopez de Prado (2014).

Canonical DSR (The Deflated Sharpe Ratio, Bailey & LdP 2014):
  SR0 = sqrt(V[SR_n]) * ((1-gamma)*Z^-1(1-1/N) + gamma*Z^-1(1-1/(N*e)))
  DSR = PSR(SR0) = Phi( ((SR_hat - SR0) * sqrt(T-1)) / sqrt(1 - g3*SR_hat + (g4-1)/4*SR_hat^2) )
where SR_hat, SR0 are NON-annualized (per-observation), g4 = raw kurtosis,
and the variance-of-trials term sqrt(V[SR_n]) SCALES the expected max.
NB: canonical denominator uses (g4-1)/4 with SR in *per-period* units;
Mertens(2002) SE^2 = (1 - g3*SR + (g4-1)/4*SR^2)/T.  (gamma = Euler-Mascheroni)
"""
import math
import numpy as np
from scipy import stats as ss
import sys
sys.path.insert(0, "/home/hashiflame/AtomiCortex")
from src.models.statistical_tests import calculate_dsr

rng = np.random.default_rng(7)
GAMMA = 0.5772156649015329

def canonical_dsr(trial_srs, T, skew, kurt_raw, n_trials):
    sr_hat = max(trial_srs)                      # per-period SR (NOT annualized)
    var_trials = np.var(trial_srs, ddof=1)
    z1 = ss.norm.ppf(1 - 1.0/n_trials)
    z2 = ss.norm.ppf(1 - 1.0/(n_trials*math.e))
    sr0 = math.sqrt(var_trials) * ((1-GAMMA)*z1 + GAMMA*z2)
    num = (sr_hat - sr0) * math.sqrt(T - 1)
    den = math.sqrt(1 - skew*sr_hat + (kurt_raw-1)/4.0*sr_hat**2)
    return ss.norm.cdf(num/den)

# --- Experiment: N strategies of pure noise, T daily returns each ---
N_TRIALS, T = 100, 1000
def one_run(true_sr_daily=0.0):
    srs_d, best_rets = [], None
    for i in range(N_TRIALS):
        r = rng.normal(true_sr_daily*0.001, 0.01, T)
        sr = r.mean()/r.std(ddof=1)
        if not srs_d or sr > max(srs_d): best_rets = r
        srs_d.append(sr)
    return srs_d, best_rets

# Case A: pure noise -> a correct DSR should be ~uniform, definitely NOT >0.95
srs_d, best = one_run(0.0)
skew = float(ss.skew(best)); kurt = float(ss.kurtosis(best, fisher=False))
dsr_canon = canonical_dsr(srs_d, T, skew, kurt, N_TRIALS)

# What the repo path does: annualized SRs + n_obs=T
srs_ann = [s*math.sqrt(365) for s in srs_d]
dsr_repo = calculate_dsr(srs_ann, n_trials=N_TRIALS, skewness=skew, kurtosis=kurt, n_obs=T)
print(f"[noise]  canonical DSR={dsr_canon:.4f}  repo DSR={dsr_repo:.4f}  best_ann_SR={max(srs_ann):.2f}")

# Case B: genuine skill, daily SR = 0.10 (annualized ~1.9, excellent real strategy)
srs_d, best = one_run(true_sr_daily=10)  # mean=0.15*sigma
skew = float(ss.skew(best)); kurt = float(ss.kurtosis(best, fisher=False))
dsr_canon = canonical_dsr(srs_d, T, skew, kurt, N_TRIALS)
srs_ann = [s*math.sqrt(365) for s in srs_d]
try:
    dsr_repo = calculate_dsr(srs_ann, n_trials=N_TRIALS, skewness=skew, kurtosis=kurt, n_obs=T)
    dsr_repo = f"{dsr_repo:.4f}"
except ValueError as e:
    dsr_repo = f"CRASH ({e}) — skew={skew:.3f} x ann_SR={max(srs_ann):.2f} makes variance term negative"
print(f"[skill]  canonical DSR={dsr_canon:.4f}  repo DSR={dsr_repo}  best_ann_SR={max(srs_ann):.2f}")

# Case C: repo PROXY path exactly as validators call it (annualized? no - WRxPFx10)
# emulate: 5 folds WR ~52-59%, PF 1.1-1.38 (real project numbers)
proxies = [(0.5324-0.5)*1.1501*10, (0.5901-0.5)*1.3786*10, (0.5594-0.5)*1.3779*10,
           (0.5248-0.5)*1.1389*10, (0.5138-0.5)*1.1464*10]
print(f"[proxy]  proxies={['%.3f'%p for p in proxies]}")
print(f"[proxy]  repo DSR (n_trials=10, no n_obs) = {calculate_dsr(proxies, n_trials=10):.4f}")
print(f"[proxy]  repo DSR (n_trials=100)          = {calculate_dsr(proxies, n_trials=100):.4f}")
