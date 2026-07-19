"""Central configuration: paths, experiment constants, analysis parameters.

Everything downstream (RCT analysis, observational demo, uplift models,
policy simulation, Streamlit app) imports from here so that a single edit
changes the whole pipeline.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------- paths -----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

RAW_FILE = DATA_RAW / "hillstrom.csv"
CLEAN_FILE = DATA_PROCESSED / "hillstrom_clean.csv"
SCORES_FILE = DATA_PROCESSED / "uplift_scores.csv"
SYNTHETIC_MARKER = DATA_RAW / "SYNTHETIC_DATA_MARKER"

DATA_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

# ---------------------------------------------------------- experiment -----
SEED = 42
TREATMENT_MAP = {
    "No E-Mail": "control",
    "Mens E-Mail": "mens",
    "Womens E-Mail": "womens",
}
ARMS = ["control", "mens", "womens"]
TREATMENTS = ["mens", "womens"]
OUTCOMES = ["visit", "conversion", "spend"]

# Part 3 focus: Women's e-mail vs control on `visit` (highest-signal cell).
UPLIFT_TREATMENT = "womens"
UPLIFT_OUTCOME = "visit"

# Model-ready feature columns produced by data_ingestion.build_clean().
FEATURES = [
    "recency",
    "history",
    "log_history",
    "history_bin",
    "mens",
    "womens",
    "newbie",
    "zip_Rural",
    "zip_Surburban",
    "zip_Urban",
    "ch_Multichannel",
    "ch_Phone",
    "ch_Web",
]

# ----------------------------------------------------- inference knobs -----
B_ATE = 2000          # bootstrap replicates for RCT ATE confidence intervals
B_OBS = 300           # bootstrap replicates for IPW (propensity re-fit inside)
B_QINI = 1000         # bootstrap replicates for Qini coefficient comparison
PS_CLIP = (0.02, 0.98)  # propensity trimming bounds for IPW / AIPW
CALIPER_SD = 0.2      # matching caliper, in SDs of the logit propensity
TEST_SIZE = 0.30
N_FOLDS = 5

# --------------------------------------------------------------- policy ----
MARGIN_PER_CONVERSION = 25.0  # contribution margin per incremental conversion ($)
COST_PER_EMAIL = 0.10         # fully loaded cost of sending one e-mail ($)

# ------------------------------------------------------------ utilities ----
SYNTHETIC_BANNER = "SYNTHETIC, regenerate on real data"


def ensure_dirs() -> None:
    """Create the data/report directories if they do not exist yet."""
    for d in (DATA_RAW, DATA_PROCESSED, REPORTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def is_synthetic() -> bool:
    """True when the pipeline is running on the synthetic fallback sample."""
    return SYNTHETIC_MARKER.exists()


def banner_md() -> str:
    """Markdown warning prepended to every artefact built from synthetic data."""
    return f"> **{SYNTHETIC_BANNER}**\n\n" if is_synthetic() else ""


def md_table(df, ndigits: int = 4) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    Small hand-rolled formatter so the project does not need an extra
    dependency (``pandas.DataFrame.to_markdown`` requires ``tabulate``).
    Very small p-values are switched to scientific notation instead of
    being rounded to 0.0000.
    """

    def fmt(v) -> str:
        if isinstance(v, float):
            if v != v:  # NaN
                return ""
            if 0 < abs(v) < 5e-4:
                return f"{v:.2e}"
            return f"{v:,.{ndigits}f}"
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"
