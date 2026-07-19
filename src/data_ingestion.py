"""Download, validate and prepare the Hillstrom e-mail RCT dataset.

The MineThatData E-Mail Analytics Challenge dataset (Hillstrom, 2008) is the
standard public uplift-modelling benchmark: 64,000 customers who purchased in
the last twelve months were randomised 1:1:1 to receive

* an e-mail featuring Men's merchandise,
* an e-mail featuring Women's merchandise, or
* no e-mail at all (control),

and were then tracked for two weeks (``visit``, ``conversion``, ``spend``).

This module

1. downloads the CSV once and caches it under ``data/raw/hillstrom.csv``,
2. validates the schema against the published one,
3. engineers a model-ready frame under ``data/processed/hillstrom_clean.csv``.

Fallback policy: if (and only if) the network makes the download impossible,
a schema-identical synthetic sample is generated instead, a marker file
``data/raw/SYNTHETIC_DATA_MARKER`` is written, and every downstream artefact
is stamped "SYNTHETIC, regenerate on real data".
"""
from __future__ import annotations

import sys
import time
import urllib.request

import numpy as np
import pandas as pd

from src import config

EXPECTED_COLUMNS = [
    "recency", "history_segment", "history", "mens", "womens",
    "zip_code", "newbie", "channel", "segment", "visit", "conversion", "spend",
]
EXPECTED_SEGMENTS = set(config.TREATMENT_MAP)
EXPECTED_ROWS = 64_000

HISTORY_SEGMENTS = [
    "1) $0 - $100", "2) $100 - $200", "3) $200 - $350",
    "4) $350 - $500", "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +",
]
ZIP_CODES = ["Surburban", "Rural", "Urban"]  # sic, spelling as shipped in the data
CHANNELS = ["Phone", "Web", "Multichannel"]


# ----------------------------------------------------------------- download
def download_raw(max_retries: int = 3, timeout: int = 120) -> bool:
    """Fetch the CSV from minethatdata.com into ``data/raw/``.

    Returns True on success. Tries the canonical http URL first, then the
    https variant, with simple exponential backoff.
    """
    urls = [config.DATA_URL, config.DATA_URL.replace("http://", "https://", 1)]
    for attempt in range(1, max_retries + 1):
        for url in urls:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "causal-uplift-marketing/1.0"}
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = resp.read()
                config.RAW_FILE.write_bytes(payload)
                print(f"[ingestion] downloaded {len(payload):,} bytes from {url}")
                return True
            except Exception as exc:  # noqa: BLE001, report and retry
                print(f"[ingestion] attempt {attempt} failed for {url}: {exc}")
        time.sleep(2**attempt)
    return False


# ----------------------------------------------------------------- validate
def validate(df: pd.DataFrame) -> None:
    """Assert the frame matches the published Hillstrom schema."""
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected columns: {list(df.columns)}")
    if set(df["segment"].unique()) != EXPECTED_SEGMENTS:
        raise ValueError(f"unexpected arms: {df['segment'].unique()}")
    if df.isna().any().any():
        raise ValueError("dataset contains missing values")
    for col in ("mens", "womens", "newbie", "visit", "conversion"):
        if not df[col].isin((0, 1)).all():
            raise ValueError(f"{col} is not binary")
    if (df["spend"] < 0).any():
        raise ValueError("negative spend values")
    if not df["recency"].between(1, 12).all():
        raise ValueError("recency outside 1-12 months")
    if len(df) != EXPECTED_ROWS:
        print(f"[ingestion] WARNING: {len(df):,} rows (published file has {EXPECTED_ROWS:,})")
    print(f"[ingestion] schema OK, {len(df):,} rows x {df.shape[1]} cols")


# ---------------------------------------------------------------- synthetic
def make_synthetic(n: int = EXPECTED_ROWS, seed: int = config.SEED) -> pd.DataFrame:
    """Schema-identical synthetic fallback (used ONLY if the download fails).

    Covariate marginals and treatment effects roughly mimic the published
    summary statistics; every downstream number produced from this sample is
    stamped "SYNTHETIC, regenerate on real data".
    """
    rng = np.random.default_rng(seed)
    seg_idx = rng.choice(len(HISTORY_SEGMENTS), n, p=[.35, .22, .17, .10, .08, .05, .03])
    lo = np.array([29, 100, 200, 350, 500, 750, 1000])[seg_idx]
    hi = np.array([100, 200, 350, 500, 750, 1000, 3345])[seg_idx]
    history = np.round(lo + (hi - lo) * rng.beta(1.3, 2.5, n), 2)
    df = pd.DataFrame({
        "recency": rng.integers(1, 13, n),
        "history_segment": np.array(HISTORY_SEGMENTS)[seg_idx],
        "history": history,
        "mens": rng.binomial(1, 0.55, n),
        "womens": rng.binomial(1, 0.55, n),
        "zip_code": rng.choice(ZIP_CODES, n, p=[.45, .15, .40]),
        "newbie": rng.binomial(1, 0.50, n),
        "channel": rng.choice(CHANNELS, n, p=[.44, .44, .12]),
        "segment": rng.choice(list(EXPECTED_SEGMENTS), n),
    })
    # Outcome model: baseline visit propensity rises with engagement; e-mails
    # add a heterogeneous lift (larger for recent, high-history customers).
    z_hist = (np.log1p(history) - np.log1p(history).mean()) / np.log1p(history).std()
    base = -2.25 + 0.35 * z_hist - 0.08 * (df["recency"] - 6)
    lift = {"No E-Mail": 0.0, "Mens E-Mail": 0.55, "Womens E-Mail": 0.40}
    eta = base + df["segment"].map(lift).to_numpy() * (1 + 0.25 * z_hist)
    p_visit = 1 / (1 + np.exp(-eta))
    df["visit"] = rng.binomial(1, np.clip(p_visit, 0, 1))
    p_conv = np.where(df["visit"] == 1, 0.06 + 0.02 * z_hist, 0.0)
    df["conversion"] = rng.binomial(1, np.clip(p_conv, 0, 1))
    df["spend"] = np.round(df["conversion"] * rng.lognormal(4.4, 0.6, n), 2)
    config.SYNTHETIC_MARKER.write_text(
        f"{config.SYNTHETIC_BANNER}\nGenerated {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"with seed={seed} because the download of\n{config.DATA_URL}\nfailed.\n"
    )
    print(f"[ingestion] WARNING: built SYNTHETIC sample ({n:,} rows), {config.SYNTHETIC_BANNER}")
    return df


# --------------------------------------------------------------------- load
def load_raw() -> pd.DataFrame:
    """Return the raw dataset, downloading (or falling back) as needed."""
    config.ensure_dirs()
    if not config.RAW_FILE.exists():
        if download_raw():
            config.SYNTHETIC_MARKER.unlink(missing_ok=True)
        else:
            df = make_synthetic()
            df.to_csv(config.RAW_FILE, index=False)
    df = pd.read_csv(config.RAW_FILE)
    validate(df)
    return df


# ----------------------------------------------------------------- features
def build_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the model-ready frame used by every downstream script.

    Adds: ``treatment`` label, ``log_history`` (spend is heavy-tailed),
    ordinal ``history_bin`` (1-7 from the segment label), and one-hot
    encodings for ``zip_code`` / ``channel``. All engineered covariates are
    pre-treatment, so they are valid adjustment variables.
    """
    out = df.copy()
    out["treatment"] = out["segment"].map(config.TREATMENT_MAP)
    out["log_history"] = np.log1p(out["history"])
    out["history_bin"] = out["history_segment"].str[0].astype(int)
    out = pd.concat(
        [
            out,
            pd.get_dummies(out["zip_code"], prefix="zip", dtype=int),
            pd.get_dummies(out["channel"], prefix="ch", dtype=int),
        ],
        axis=1,
    )
    missing = [c for c in config.FEATURES if c not in out.columns]
    if missing:
        raise ValueError(f"feature engineering failed, missing: {missing}")
    return out


def load_clean(rebuild: bool = False) -> pd.DataFrame:
    """Load the processed frame, building it (and the raw cache) if needed."""
    if config.CLEAN_FILE.exists() and not rebuild:
        return pd.read_csv(config.CLEAN_FILE)
    clean = build_clean(load_raw())
    clean.to_csv(config.CLEAN_FILE, index=False)
    print(f"[ingestion] wrote {config.CLEAN_FILE.relative_to(config.PROJECT_ROOT)}")
    return clean


def main() -> int:
    df = load_clean(rebuild=True)
    print("\narm sizes:")
    print(df["treatment"].value_counts().to_string())
    print("\nmean outcomes by arm:")
    print(df.groupby("treatment")[config.OUTCOMES].mean().round(4).to_string())
    if config.is_synthetic():
        print(f"\n*** {config.SYNTHETIC_BANNER} ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
