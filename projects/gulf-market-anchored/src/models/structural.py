"""Football baselines（§4.3）：Elo / Independent Poisson / Dixon–Coles。

implementation note，implementation note：
- Elo：Elo_0=1500、K=20 implementation note；ΔElo implementation note home_adv；h implementation note update implementation note
  expected-score implementation note；implementation note temporal origin implementation note（fit-at-origin）。
- Independent Poisson：sum-to-zero attack/defence、ridge-penalized MLE、
  cold-start（implementation note α=0,δ=0，implementation note home_adv）。
- Dixon–Coles：implementation note ρ、temporal decay ξ（grid）、deterministic bounded
  multi-start，implementation note fail loudly（implementation note Poisson）。

implementation note (N,3)，implementation note [A, D, H]（= class_order）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from sklearn.linear_model import LogisticRegression

from .. import metrics
from ..pooled_split import temporal_batches, match_time

ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_H_GRID = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]

# Dixon–Coles implementation note decay grid（day⁻¹）
DC_XI_GRID = [0.0, 0.0005, 0.001, 0.0015, 0.002]

# implementation note
GOAL_GRID_MAX = 20
EPS = 1e-12

# Poisson/DC ridge-penalized MLE implementation note（implementation note，implementation note tune）
RIDGE_ALPHA = 0.01

# Dixon–Coles low-score correction positivity margin.  Fixed before definitive
# evaluation as a numerical stability constraint: every training-fixture tau term must
# exceed this margin; predictions that still violate probability validity fail loudly.
TAU_EPS = 0.05


# ------------------------------------------------------------------ #
# implementation note
# ------------------------------------------------------------------ #
def _home_key_col(df: pd.DataFrame) -> str:
    return "home_team_key" if "home_team_key" in df.columns else "home_team_name"


def _away_key_col(df: pd.DataFrame) -> str:
    return "away_team_key" if "away_team_key" in df.columns else "away_team_name"


def _match_records(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize completed fixtures while preserving the authoritative batch metadata.

    `match_datetime`, optional `date`, and `timestamp_precision` are retained so Elo's
    historical forward pass can apply exactly the same equal-time/date-only batching rule
    as test-time online prediction.  Poisson/DC ignore the extra columns.
    """
    hk, ak = _home_key_col(df), _away_key_col(df)
    dt = pd.to_datetime(df["match_datetime"], errors="coerce", utc=True)
    if "date" in df.columns:
        date = pd.to_datetime(df["date"], errors="coerce", utc=True)
        dt = dt.fillna(date)
    rec = pd.DataFrame({
        "match_datetime": dt,
        "date": df["date"].to_numpy() if "date" in df.columns else dt.dt.strftime("%Y-%m-%d").to_numpy(),
        "timestamp_precision": (df["timestamp_precision"].astype(str).to_numpy()
                                if "timestamp_precision" in df.columns else np.repeat("exact", len(df))),
        "time": dt,
        "home": df[hk].astype(str),
        "away": df[ak].astype(str),
        "hg": pd.to_numeric(df["home_team_goal_count"], errors="coerce").astype(float),
        "ag": pd.to_numeric(df["away_team_goal_count"], errors="coerce").astype(float),
    }).dropna(subset=["match_datetime", "home", "away", "hg", "ag"])
    rec["result"] = np.where(rec["hg"] > rec["ag"], "H",
                    np.where(rec["hg"] < rec["ag"], "A", "D"))
    return rec.sort_values("match_datetime").reset_index(drop=True)


def _fixture_records(df: pd.DataFrame) -> pd.DataFrame:
    """implementation note（§4.3）：implementation note (time, home, away)，implementation note FT implementation note/implementation note。

    implementation note _match_records（implementation note，implementation note hg/ag/result）implementation note。implementation note probe（elo/poisson/dc
    implementation note *_proba）implementation note home/away implementation note，implementation note FT outcome implementation note。
    """
    hk, ak = _home_key_col(df), _away_key_col(df)
    rec = pd.DataFrame({
        "time": pd.to_datetime(df["match_datetime"], errors="coerce", utc=True),
        "home": df[hk].astype(str),
        "away": df[ak].astype(str),
    }).dropna(subset=["home", "away"])
    return rec.reset_index(drop=True)


def _prior_rating_series(rec: pd.DataFrame, h: float) -> tuple[dict, np.ndarray]:
    """Run the historical Elo forward pass with the frozen simultaneous-fixture rule.

    Every fixture in one exact-timestamp/date-only-day batch receives a pre-match rating
    from the same rating snapshot; only after all batch predictions are formed are any
    outcomes consumed.  A team appearing twice in one batch is treated as a data error.
    """
    ratings: dict[str, float] = {}
    d_elo = np.zeros(len(rec))
    for pos in temporal_batches(rec):
        teams = []
        snapshot = {}
        for i in pos:
            row = rec.iloc[int(i)]
            home, away = str(row["home"]), str(row["away"])
            teams.extend([home, away])
            rh = ratings.get(home, ELO_INITIAL)
            ra = ratings.get(away, ELO_INITIAL)
            snapshot[int(i)] = (rh, ra)
            d_elo[int(i)] = rh - ra
        if len(teams) != len(set(teams)):
            raise RuntimeError("Elo batch contains a team more than once; chronology is ambiguous")
        for i in pos:
            row = rec.iloc[int(i)]
            home, away = str(row["home"]), str(row["away"])
            rh, ra = snapshot[int(i)]
            e_home = 1.0 / (1.0 + 10.0 ** (-(rh + h - ra) / 400.0))
            if row["hg"] > row["ag"]:
                s_home = 1.0
            elif row["hg"] < row["ag"]:
                s_home = 0.0
            else:
                s_home = 0.5
            ratings[home] = rh + ELO_K * (s_home - e_home)
            ratings[away] = ra + ELO_K * ((1.0 - s_home) - (1.0 - e_home))
    return ratings, d_elo


@dataclass
class EloModel:
    h: float
    calibration: LogisticRegression
    ratings: dict[str, float]
    origin_max_time: pd.Timestamp
    trained_match_count: int

    def proba(self, rec: pd.DataFrame) -> np.ndarray:
        """implementation note rec（implementation note home/away）implementation note (N,3) implementation note，implementation note origin implementation note。"""
        d_elo = np.array([
            self.ratings.get(row["home"], ELO_INITIAL)
            - self.ratings.get(row["away"], ELO_INITIAL)
            for _, row in rec.iterrows()
        ])
        p = self.calibration.predict_proba(d_elo.reshape(-1, 1))
        # sklearn implementation note classes_；implementation note [A,D,H]
        order = [list(self.calibration.classes_).index(c) for c in metrics.class_order()]
        return p[:, order]


def fit_elo_fixed_h(train_df: pd.DataFrame, h: float) -> EloModel:
    """Fit Elo ratings/calibration for a pre-specified home-advantage value h."""
    rec=_match_records(train_df)
    if len(rec)<20:
        raise ValueError(f"Elo training sample too small ({len(rec)})")
    ratings,d_elo=_prior_rating_series(rec,float(h))
    clf=LogisticRegression(max_iter=2000)
    clf.fit(d_elo.reshape(-1,1),rec["result"])
    return EloModel(h=float(h),calibration=clf,ratings=ratings,origin_max_time=rec["time"].max(),trained_match_count=len(rec))


def _clf_proba(clf: LogisticRegression, d_elo: np.ndarray) -> np.ndarray:
    p = clf.predict_proba(d_elo.reshape(-1, 1))
    order = [list(clf.classes_).index(c) for c in metrics.class_order()]
    return p[:, order]


def elo_proba(model: EloModel, test_df: pd.DataFrame) -> np.ndarray:
    rec = _fixture_records(test_df)
    return model.proba(rec)


def elo_proba_online(model: EloModel, test_df: pd.DataFrame) -> np.ndarray:
    """Prequential Elo with exact-timestamp and date-only day batching."""
    te=test_df.reset_index(drop=True); hk,ak=_home_key_col(te),_away_key_col(te)
    home=te[hk].astype(str).to_numpy(); away=te[ak].astype(str).to_numpy()
    hg=pd.to_numeric(te["home_team_goal_count"],errors="coerce").to_numpy(float); ag=pd.to_numeric(te["away_team_goal_count"],errors="coerce").to_numpy(float)
    ratings=dict(model.ratings); probs=np.zeros((len(te),3))
    for pos in temporal_batches(te):
        for i in pos:
            rh=ratings.get(home[i],ELO_INITIAL); ra=ratings.get(away[i],ELO_INITIAL); probs[i]=_clf_proba(model.calibration,np.array([rh-ra]))[0]
        for i in pos:
            rh=ratings.get(home[i],ELO_INITIAL); ra=ratings.get(away[i],ELO_INITIAL)
            if np.isfinite(hg[i]) and np.isfinite(ag[i]):
                e=1/(1+10**(-(rh+model.h-ra)/400.0)); sh=1.0 if hg[i]>ag[i] else (0.0 if hg[i]<ag[i] else 0.5)
                ratings[home[i]]=rh+ELO_K*(sh-e); ratings[away[i]]=ra+ELO_K*((1-sh)-(1-e))
    metrics.assert_valid_proba(probs,context="Elo online")
    return probs


# ------------------------------------------------------------------ #
# Poisson / Dixon–Coles implementation note
# ------------------------------------------------------------------ #
def _poisson_grid(lam: float, max_k: int = GOAL_GRID_MAX) -> np.ndarray:
    """implementation note 0..max_k implementation note Poisson pmf（implementation note K=GOAL_GRID_MAX=20，implementation note）。

    λ≤0 implementation note clip implementation note 1e-6 implementation note log/exp implementation note。implementation note λ（implementation note <4），K=20 implementation note
    implementation note <1e-8、implementation note，implementation note（implementation note K implementation note）。
    """
    if lam <= 0:
        lam = 1e-6
    k = np.arange(max_k + 1)
    logp = k * np.log(lam) - lam - np.array([np.sum(np.log(np.arange(1, kk + 1))) if kk > 0 else 0.0 for kk in k])
    p = np.exp(logp)
    return p / p.sum()  # implementation note（implementation note）


def _goal_proba_matrix(lam_h: float, lam_a: float) -> np.ndarray:
    ph = _poisson_grid(lam_h)
    pa = _poisson_grid(lam_a)
    return np.outer(ph, pa)  # [i implementation note, j implementation note]


def _three_class_from_matrix(M: np.ndarray) -> np.ndarray:
    """implementation note goals implementation note（M[i,j]=P(implementation note i implementation note,implementation note j implementation note)）implementation note [P(A), P(D), P(H)]。

    implementation note i>j（implementation note，excl implementation note）；implementation note i<j（implementation note，excl implementation note）；implementation note i==j（implementation note）。
    """
    n = M.shape[0]
    p_home = M[np.tril_indices(n, k=-1)].sum()
    p_away = M[np.triu_indices(n, k=1)].sum()
    p_draw = np.trace(M)
    v = np.array([p_away, p_draw, p_home])
    return v / v.sum()


def _team_index_maps(rec: pd.DataFrame) -> tuple[dict, dict, list]:
    teams = sorted(set(rec["home"]) | set(rec["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    return idx, idx, teams


def _team_index_arrays(rec: pd.DataFrame, idx: dict) -> tuple[np.ndarray, np.ndarray]:
    """home/away implementation note（implementation note）。"""
    hi = rec["home"].map(idx).to_numpy(dtype=int)
    ai = rec["away"].map(idx).to_numpy(dtype=int)
    return hi, ai


# implementation note attack/defence implementation note（implementation note sum-to-zero）
def _param_vec(att_free: np.ndarray, def_free: np.ndarray) -> np.ndarray:
    att = np.append(att_free, -att_free.sum())
    defe = np.append(def_free, -def_free.sum())
    return att, defe


@dataclass
class PoissonModel:
    mu: float
    home_adv: float
    attack: dict[str, float]
    defence: dict[str, float]
    teams: list[str]
    origin_max_time: pd.Timestamp
    trained_match_count: int

    def lambda_home_away(self, rec: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lh, la = [], []
        for _, row in rec.iterrows():
            a_h = self.attack.get(row["home"], 0.0)
            d_h = self.defence.get(row["home"], 0.0)
            a_a = self.attack.get(row["away"], 0.0)
            d_a = self.defence.get(row["away"], 0.0)
            lh.append(np.exp(self.mu + self.home_adv + a_h + d_a))
            la.append(np.exp(self.mu + a_a + d_h))
        return np.array(lh), np.array(la)

    def proba(self, rec: pd.DataFrame) -> np.ndarray:
        lh, la = self.lambda_home_away(rec)
        out = np.zeros((len(rec), 3))
        for i in range(len(rec)):
            M = _goal_proba_matrix(lh[i], la[i])
            out[i] = _three_class_from_matrix(M)
        return out


def _poisson_nll(theta: np.ndarray, hi: np.ndarray, ai: np.ndarray,
                 hg: np.ndarray, ag: np.ndarray, T: int,
                 weights: np.ndarray | None = None) -> float:
    mu, home_adv = theta[0], theta[1]
    att_free = theta[2:2 + T - 1]
    def_free = theta[2 + T - 1:2 + 2 * (T - 1)]
    att, defe = _param_vec(att_free, def_free)
    lh = mu + home_adv + att[hi] + defe[ai]
    la = mu + att[ai] + defe[hi]
    ll = (-np.exp(lh) + hg * lh - gammaln(hg + 1.0)) \
       + (-np.exp(la) + ag * la - gammaln(ag + 1.0))
    ll = ll.sum() if weights is None else (ll * weights).sum()
    pen = RIDGE_ALPHA * (np.sum(att ** 2) + np.sum(defe ** 2))
    return -ll + pen


def fit_poisson(train_df: pd.DataFrame) -> PoissonModel:
    rec = _match_records(train_df)
    idx, _, teams = _team_index_maps(rec)
    T = len(teams)
    if T < 2:
        raise ValueError("Poisson implementation note 2 implementation note")

    hi, ai = _team_index_arrays(rec, idx)
    hg = rec["hg"].to_numpy(dtype=float)
    ag = rec["ag"].to_numpy(dtype=float)

    mu0 = np.log(max(0.5, (hg.mean() + ag.mean()) / 2.0))
    x0 = np.concatenate([[mu0, 0.1], np.zeros(T - 1), np.zeros(T - 1)])

    res = minimize(
        _poisson_nll, x0, args=(hi, ai, hg, ag, T, None),
        method="L-BFGS-B",
        options={"maxiter": 10000, "maxfun": 200000, "ftol": 1e-10, "gtol": 1e-8, "maxls": 50},
    )
    if not np.isfinite(res.fun):
        raise RuntimeError("Poisson MLE implementation note（implementation note）")
    if not res.success:
        raise RuntimeError(f"Poisson MLE implementation note: {res.message}（status={res.status}）")

    musucc, home_adv = res.x[0], res.x[1]
    att_free = res.x[2:2 + T - 1]
    def_free = res.x[2 + T - 1:2 + 2 * (T - 1)]
    att, defe = _param_vec(att_free, def_free)

    return PoissonModel(
        mu=musucc,
        home_adv=home_adv,
        attack=dict(zip(teams, att)),
        defence=dict(zip(teams, defe)),
        teams=teams,
        origin_max_time=rec["time"].max(),
        trained_match_count=len(rec),
    )


def poisson_proba(model: PoissonModel, test_df: pd.DataFrame) -> np.ndarray:
    rec = _fixture_records(test_df)
    return model.proba(rec)


# ------------------------------------------------------------------ #
# Dixon–Coles
# ------------------------------------------------------------------ #
def _dc_goal_proba_matrix(lam_h: float, lam_a: float, rho: float,
                          strict: bool = True) -> np.ndarray:
    ph = _poisson_grid(lam_h)
    pa = _poisson_grid(lam_a)
    M = np.outer(ph, pa)
    # implementation note（implementation note cell）
    M[0, 0] *= (1.0 - lam_h * lam_a * rho)
    M[0, 1] *= (1.0 + lam_h * rho)
    M[1, 0] *= (1.0 + lam_a * rho)
    M[1, 1] *= (1.0 - rho)
    # P0 implementation note：implementation note cell implementation note（implementation note λ_H·ρ < -1 implementation note τ(0,1)<0）。
    #   strict=True  （implementation note/implementation note）：fail loudly，implementation note clip implementation note 0 implementation note。
    #   strict=False （implementation note）   ：implementation note fixture implementation note λ implementation note，implementation note
    #     implementation note——implementation note NaN implementation note，implementation note fail loudly，implementation note clip、implementation note。
    if np.any(M < 0.0):
        if strict:
            raise RuntimeError(
                f"Dixon–Coles implementation note（lam_h={lam_h:.4f}, lam_a={lam_a:.4f}, "
                f"rho={rho:.4f}），implementation note，implementation note clip")
        return np.full_like(M, np.nan)
    s = M.sum()
    if not np.isfinite(s) or s <= 0:
        if strict:
            raise RuntimeError("Dixon–Coles implementation note（implementation note 0）")
        return np.full_like(M, np.nan)
    return M / s


def _dc_neg_loglik(params: np.ndarray, hi: np.ndarray, ai: np.ndarray,
                   hg: np.ndarray, ag: np.ndarray, T: int,
                   weights: np.ndarray) -> float:
    """implementation note（implementation note decay implementation note）；ρ implementation note params[-1]。

    DC pmf = τ(hg,ag) * Pois(hg;λ_h) * Pois(ag;λ_a)；τ implementation note cell implementation note。
    implementation note τ implementation note**implementation note fixture** implementation note > TAU_EPS（implementation note），
    implementation note fixture implementation note score cell。implementation note fixture implementation note τ ≤ TAU_EPS implementation note
    implementation note cell implementation note 0/implementation note；implementation note（implementation note）
    implementation note，implementation note log-rate implementation note exp implementation note NaN（implementation note、
    implementation note）。implementation note clip / implementation note。
    """
    mu, home_adv, rho = params[0], params[1], params[-1]
    att_free = params[2:2 + T - 1]
    def_free = params[2 + T - 1:2 + 2 * (T - 1)]
    att, defe = _param_vec(att_free, def_free)

    # implementation note：implementation note log-rate implementation note [-20,20]（λ ∈ (e^-20, e^20)）implementation note exp implementation note。implementation note λ
    # implementation note e^20，implementation note；implementation note/NaN。
    lr_h = mu + home_adv + att[hi] + defe[ai]
    lr_a = mu + att[ai] + defe[hi]
    lam_h = np.exp(np.clip(lr_h, -20.0, 20.0))
    lam_a = np.exp(np.clip(lr_a, -20.0, 20.0))

    tau00 = 1.0 - lam_h * lam_a * rho
    tau01 = 1.0 + lam_h * rho
    tau10 = 1.0 + lam_a * rho
    tau11 = 1.0 - rho  # implementation note

    # implementation note：implementation note fixture implementation note τ implementation note ≤ TAU_EPS ⇒ implementation note cell implementation note。
    # TAU_EPS implementation note，implementation note outer-test implementation note。
    if (bool(np.any(tau00 <= TAU_EPS)) or bool(np.any(tau01 <= TAU_EPS))
            or bool(np.any(tau10 <= TAU_EPS)) or bool(tau11 <= TAU_EPS)):
        return 1e12

    logp = (hg * np.log(lam_h) - lam_h - gammaln(hg + 1.0)) \
         + (ag * np.log(lam_a) - lam_a - gammaln(ag + 1.0))

    tau = np.ones_like(lam_h)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = tau00[m00]
    tau[m01] = tau01[m01]
    tau[m10] = tau10[m10]
    tau[m11] = tau11

    ll = logp + np.log(tau)
    ll = ll.sum() if weights is None else (ll * weights).sum()
    pen = RIDGE_ALPHA * (np.sum(att ** 2) + np.sum(defe ** 2))
    return -ll + pen


def _dc_params_valid(theta: np.ndarray, hi: np.ndarray, ai: np.ndarray,
                     T: int, eps: float = TAU_EPS) -> bool:
    """implementation note DC implementation note θ，implementation note training fixture implementation note τ implementation note > eps（implementation note）。

    implementation note multi-start implementation note：implementation note，implementation note（P0 implementation note）。
    """
    mu, home_adv, rho = theta[0], theta[1], theta[-1]
    att_free = theta[2:2 + T - 1]
    def_free = theta[2 + T - 1:2 + 2 * (T - 1)]
    att, defe = _param_vec(att_free, def_free)
    # implementation note _dc_neg_loglik implementation note log-rate，implementation note/implementation note。
    lr_h = mu + home_adv + att[hi] + defe[ai]
    lr_a = mu + att[ai] + defe[hi]
    lam_h = np.exp(np.clip(lr_h, -20.0, 20.0))
    lam_a = np.exp(np.clip(lr_a, -20.0, 20.0))
    return (bool(np.all(1.0 - lam_h * lam_a * rho > eps))
            and bool(np.all(1.0 + lam_h * rho > eps))
            and bool(np.all(1.0 + lam_a * rho > eps))
            and bool(1.0 - rho > eps))


# Dixon–Coles predetermined implementation note（deterministic bounded multi-start）
_DC_INIT_RHO = [0.0, -0.05, 0.05, 0.10]
_DC_RESTARTS = len(_DC_INIT_RHO)


@dataclass
class DixonColesModel:
    mu: float
    home_adv: float
    rho: float
    xi: float
    attack: dict[str, float]
    defence: dict[str, float]
    teams: list[str]
    origin_max_time: pd.Timestamp
    trained_match_count: int

    def proba(self, rec: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(rec), 3))
        for i, row in enumerate(rec.itertuples()):
            lam_h = np.exp(self.mu + self.home_adv + self.attack.get(row.home, 0.0)
                           + self.defence.get(row.away, 0.0))
            lam_a = np.exp(self.mu + self.attack.get(row.away, 0.0)
                           + self.defence.get(row.home, 0.0))
            # strict=False：implementation note fixture implementation note λ implementation note NaN implementation note（implementation note cliplow）。
            M = _dc_goal_proba_matrix(lam_h, lam_a, self.rho, strict=False)
            out[i] = _three_class_from_matrix(M)
        return out


def fit_dixon_coles(train_df: pd.DataFrame, xi: float) -> DixonColesModel:
    """deterministic bounded multi-start；implementation note fail loudly（implementation note）。"""
    rec = _match_records(train_df)
    idx, _, teams = _team_index_maps(rec)
    T = len(teams)
    if T < 2:
        raise ValueError("Dixon–Coles implementation note 2 implementation note")

    hi, ai = _team_index_arrays(rec, idx)
    hg = rec["hg"].to_numpy(dtype=float)
    ag = rec["ag"].to_numpy(dtype=float)

    # temporal decay implementation note（implementation note）
    tmax = rec["time"].max()
    days = (tmax - rec["time"]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    weights = np.exp(-xi * days)

    mu0 = np.log(max(0.5, (rec["hg"].mean() + rec["ag"].mean()) / 2.0))

    best: None | tuple[float, np.ndarray] = None
    for rho0 in _DC_INIT_RHO:
        x0 = np.concatenate([[mu0, 0.1], np.zeros(T - 1), np.zeros(T - 1), [rho0]])
        # implementation note：ρ implementation note（implementation note [-1, 1] implementation note），implementation note
        bounds = [(None, None)] * (2 + 2 * (T - 1)) + [(-0.5, 0.5)]
        res = minimize(
            _dc_neg_loglik, x0, args=(hi, ai, hg, ag, T, weights),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 10000, "maxfun": 200000, "ftol": 1e-10, "gtol": 1e-8, "maxls": 50},
        )
        if not np.isfinite(res.fun):
            continue
        if not res.success:
            continue  # implementation note restart implementation note，implementation note；implementation note fail-loud
        # P0 implementation note：implementation note training fixture implementation note τ implementation note > 0；
        # implementation note（implementation note τ≤0）implementation note res.success implementation note，implementation note clip。
        if not _dc_params_valid(res.x, hi, ai, T):
            continue
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x)

    if best is None:
        # fail loudly —— implementation note Poisson
        raise RuntimeError(
            "Dixon–Coles：implementation note（implementation note/implementation note），"
            "implementation note，implementation note DC headline result"
        )

    xstar = best[1]
    musucc, home_adv, rho = xstar[0], xstar[1], xstar[-1]
    att_free = xstar[2:2 + T - 1]
    def_free = xstar[2 + T - 1:2 + 2 * (T - 1)]
    att, defe = _param_vec(att_free, def_free)

    model = DixonColesModel(
        mu=musucc, home_adv=home_adv, rho=rho, xi=xi,
        attack=dict(zip(teams, att)), defence=dict(zip(teams, defe)),
        teams=teams, origin_max_time=rec["time"].max(), trained_match_count=len(rec),
    )
    # implementation note：implementation note
    metrics.assert_valid_proba(model.proba(rec), context="Dixon–Coles")
    return model


def dc_proba(model: DixonColesModel, test_df: pd.DataFrame) -> np.ndarray:
    rec = _fixture_records(test_df)
    p = model.proba(rec)
    metrics.assert_valid_proba(p, context="Dixon-Coles prediction")
    return p


# ------------------------------------------------------------------ #
# exact-batch prequential rolling re-fit
# ------------------------------------------------------------------ #
def _rolling_proba(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   fit_fn: Callable, proba_fn: Callable, xi: float | None = None) -> np.ndarray:
    """Refit before each legal timestamp/day batch. All fixtures in a batch are
    predicted before any result from that batch is appended to history.
    """
    te=test_df.reset_index(drop=True).copy(); history=train_df.reset_index(drop=True).copy()
    out=np.full((len(te),3),np.nan)
    for pos in temporal_batches(te):
        batch=te.iloc[pos].copy()
        model=fit_fn(history) if xi is None else fit_fn(history,xi=xi)
        p=proba_fn(model,batch)
        metrics.assert_valid_proba(p, context="rolling structural")
        if len(p)!=len(pos): raise RuntimeError("rolling batch prediction length mismatch")
        out[pos]=p
        history=pd.concat([history,batch],ignore_index=True)
    metrics.assert_valid_proba(out, context="rolling structural complete")
    return out

def poisson_proba_rolling(train_df: pd.DataFrame, test_df: pd.DataFrame, freq: str | None=None) -> np.ndarray:
    return _rolling_proba(train_df,test_df,fit_poisson,poisson_proba)

def dc_proba_rolling(train_df: pd.DataFrame, test_df: pd.DataFrame, xi: float, freq: str | None=None) -> np.ndarray:
    return _rolling_proba(train_df,test_df,fit_dixon_coles,dc_proba,xi=xi)


# ------------------------------------------------------------------ #
# Conditional Poisson（implementation note baseline，§6.5）
# ------------------------------------------------------------------ #
def _three_class_given_ht(lam_h2: float, lam_a2: float, h_ht: int, a_ht: int) -> np.ndarray:
    """P(FT outcome | implementation note HT implementation note (h_ht,a_ht)) = implementation note 2nd-half Poisson implementation note
    implementation note / implementation note / implementation note，implementation note [P(A), P(D), P(H)]。

    implementation note（§6.5，implementation note baseline）：
    - implementation note ~ Pois(r·λ_h) implementation note Pois(r·λ_a)，implementation note；λ implementation note pre-match Independent
      Poisson implementation note（implementation note home_adv），r = second_half_ratio implementation note「implementation note」，
      implementation note outer-train implementation note（implementation note 0.5，implementation note fit_conditional_poisson）。
    - implementation note = HT implementation note + implementation note。
    - implementation note Dixon–Coles implementation note ρ（implementation note baseline implementation note）。
    """
    ph = _poisson_grid(lam_h2)
    pa = _poisson_grid(lam_a2)
    p_home = 0.0
    p_away = 0.0
    p_draw = 0.0
    for i in range(len(ph)):
        for j in range(len(pa)):
            fh = h_ht + i
            fa = a_ht + j
            m = ph[i] * pa[j]
            if fh > fa:
                p_home += m
            elif fh < fa:
                p_away += m
            else:
                p_draw += m
    v = np.array([p_away, p_draw, p_home])
    v = np.clip(v, EPS, None)
    return v / v.sum()


@dataclass
class ConditionalPoissonModel:
    base: PoissonModel
    second_half_ratio: float = 0.5

    def proba(self, ht_df: pd.DataFrame) -> np.ndarray:
        """implementation note ht_df（implementation note HT implementation note，implementation note T_seq implementation note）implementation note (N,3) implementation note。"""
        hk, ak = _home_key_col(ht_df), _away_key_col(ht_df)
        rec = pd.DataFrame({"home": ht_df[hk].astype(str), "away": ht_df[ak].astype(str)})
        lh, la = self.base.lambda_home_away(rec)  # implementation note λ，implementation note（implementation note）
        lh2 = lh * self.second_half_ratio
        la2 = la * self.second_half_ratio

        h_ht = pd.to_numeric(ht_df["home_team_goal_count_half_time"], errors="coerce").to_numpy(dtype=float)
        a_ht = pd.to_numeric(ht_df["away_team_goal_count_half_time"], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(h_ht)) or np.any(~np.isfinite(a_ht)):
            raise ValueError("Conditional Poisson implementation note HT implementation note（implementation note T_seq implementation note）")

        out = np.zeros((len(rec), 3), dtype=float)
        for i in range(len(rec)):
            out[i] = _three_class_given_ht(lh2[i], la2[i], int(h_ht[i]), int(a_ht[i]))
        return out


def _estimate_second_half_ratio(df: pd.DataFrame) -> float:
    """implementation note outer-train implementation note second-half implementation note r = Σ(implementation note)/Σ(implementation note)。

    implementation note FT implementation note HT implementation note（implementation note、implementation note），implementation note 0.5 implementation note「implementation note
    implementation note」implementation note。implementation note 0（implementation note）implementation note 0.5。
    """
    hg = pd.to_numeric(df["home_team_goal_count"], errors="coerce")
    ag = pd.to_numeric(df["away_team_goal_count"], errors="coerce")
    hht = pd.to_numeric(df["home_team_goal_count_half_time"], errors="coerce")
    aht = pd.to_numeric(df["away_team_goal_count_half_time"], errors="coerce")
    valid = hg.notna() & ag.notna() & hht.notna() & aht.notna()
    second = (hg[valid] - hht[valid]) + (ag[valid] - aht[valid])
    full = hg[valid] + ag[valid]
    denom = float(full.sum())
    if denom <= 0:
        return 0.5
    return float(np.clip(second.sum() / denom, 0.0, 1.0))


def fit_conditional_poisson(train_df: pd.DataFrame) -> ConditionalPoissonModel:
    """pre-match Independent Poisson implementation note（implementation note outer-train implementation note fit，implementation note §4.3 implementation note），
    implementation note half-time implementation note Poisson；second-half implementation note outer-train implementation note。"""
    base = fit_poisson(train_df)
    ratio = _estimate_second_half_ratio(train_df)
    return ConditionalPoissonModel(base=base, second_half_ratio=ratio)


def conditional_poisson_proba(model: ConditionalPoissonModel, ht_df: pd.DataFrame) -> np.ndarray:
    return model.proba(ht_df)

def conditional_poisson_proba_rolling(train_df: pd.DataFrame, query_df: pd.DataFrame,
                                        chronology_df: pd.DataFrame | None = None) -> np.ndarray:
    """League-specific prequential conditional-Poisson predictions.

    `query_df` contains the T_seq fixtures that need a prediction.  `chronology_df` may
    contain every fixture in the test period; all completed batches are appended to the
    history even when they are not T_seq, so later structural predictions can legally use
    them.  Same-time/date-only batches are never consumed within-batch.
    """
    q=query_df.reset_index(drop=True).copy()
    chrono=(query_df if chronology_df is None else chronology_df).reset_index(drop=True).copy()
    if "fixture_key" not in q.columns or "fixture_key" not in chrono.columns:
        raise ValueError("conditional Poisson rolling requires fixture_key")
    qkeys=q["fixture_key"].astype(str).tolist()
    qpos={k:i for i,k in enumerate(qkeys)}
    if len(qpos)!=len(qkeys):
        raise ValueError("conditional Poisson query fixture_key must be unique")
    out=np.full((len(q),3),np.nan)
    history=train_df.reset_index(drop=True).copy()
    for pos in temporal_batches(chrono):
        batch=chrono.iloc[pos].copy()
        keys=batch["fixture_key"].astype(str).tolist()
        take=[j for j,k in enumerate(keys) if k in qpos]
        if take:
            model=fit_conditional_poisson(history)
            qb=batch.iloc[take].reset_index(drop=True)
            p=model.proba(qb)
            metrics.assert_valid_proba(p,context="Conditional Poisson rolling")
            for r,k in enumerate(qb["fixture_key"].astype(str)):
                out[qpos[k]]=p[r]
        history=pd.concat([history,batch],ignore_index=True)
    if not np.isfinite(out).all():
        missing=[qkeys[i] for i in np.flatnonzero(~np.isfinite(out).all(axis=1))[:10]]
        raise RuntimeError(f"Conditional Poisson rolling missed query fixtures: {missing}")
    metrics.assert_valid_proba(out,context="Conditional Poisson rolling complete")
    return out
