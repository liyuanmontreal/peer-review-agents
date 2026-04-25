import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env():
    """Load .env file from project root (if present)."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_file)

def get_credentials() -> tuple[str, str]:
    """Return (username, password) from environment variables.

    Raises KeyError if either variable is unset or empty.
    """
    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        raise KeyError(
            "OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD must be set in the "
            "environment (or in a .env file)."
        )
    return username, password

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_V2_BASE_URL = "https://api2.openreview.net"

DEFAULT_LLM_MODEL = "gpt-4o-mini"


def get_llm_model() -> str:
    """Return the LLM model ID from ``GSR_LLM_MODEL`` or the default."""
    return os.environ.get("GSR_LLM_MODEL", DEFAULT_LLM_MODEL)

DEFAULT_VENUE_ID = "ICLR.cc/2025/Conference"

INVITATION_SUFFIXES: dict[str, list[str]] = {
    "review": ["Official_Review"],
    "rebuttal": [
        "Official_Comment",   # ICLR, NeurIPS
        "Rebuttal",           # ICML
        "Rebuttal_Comment",   # ICML (reviewer reply to rebuttal)
        "Reply_Rebuttal_Comment",  # ICML (author reply to reviewer reply)
    ],
    "meta_review": ["Meta_Review"],
    "decision": ["Decision"],
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root

def _resolve_dir(value: str | Path) -> Path:
    """Resolve a directory path.

    - If ``value`` is absolute, return it as-is (resolved).
    - If relative, interpret it relative to the repo root.
    """
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


# # ---------------------------------------------------------------------------
# # Dataset vs Workspace
# # ---------------------------------------------------------------------------
DATASET_DIR = _resolve_dir(os.getenv("GSR_DATASET_DIR", "data"))
DEFAULT_WORKSPACE_DIR = PROJECT_ROOT / "workspace"

def get_workspace_info() -> tuple[Path, str]:
    """
    Returns (workspace_dir, source_label)
    source_label is only for printing/debugging.
    """
    ws = os.environ.get("GSR_WORKSPACE")
    src = os.environ.get("GSR_WORKSPACE_SOURCE", "default")
    if ws:
        return Path(ws).resolve(), src
    return DEFAULT_WORKSPACE_DIR.resolve(), "default"

# Workspace / DB paths (workspace-aware)
WORKSPACE_DIR, _WORKSPACE_SRC = get_workspace_info()

# Runtime artifact paths (always under WORKSPACE_DIR)
JSON_DIR = WORKSPACE_DIR / "json"
CSV_DIR = WORKSPACE_DIR / "csv"
DB_PATH = WORKSPACE_DIR / "gsr.db"
REPORT_DIR = WORKSPACE_DIR / "reports"

# Example other dirs (optional, but recommended to keep artifacts under workspace)
PDF_DIR = WORKSPACE_DIR / "pdf"
OUTPUT_DIR = WORKSPACE_DIR / "output"

def ensure_workspace_dirs() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

EDA_RAW_FIG_DIR = OUTPUT_DIR / "eda" / "raw_data"
EDA_PROCESSED_FIG_DIR = OUTPUT_DIR / "eda" / "processed_data"

def get_workspace_info() -> tuple[Path, str]:
    """Return (workspace_dir, source)."""
    return WORKSPACE_DIR, _WORKSPACE_SRC
