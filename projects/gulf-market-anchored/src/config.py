from __future__ import annotations
import json, os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "official_data"
OUTPUT_DIR = PROJECT_ROOT / "results"
TABLE_DIR = PROJECT_ROOT / "tables"

def load_json(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def protocol() -> dict:
    return load_json("protocol_final.json")

def seed() -> int:
    return int(protocol()["random_seed"])

def assert_definitive_allowed(confirm: bool) -> None:
    env_name = protocol().get("definitive_lock_env", "ALLOW_DEFINITIVE_OUTER_TEST")
    if not confirm or os.getenv(env_name) != "1":
        raise RuntimeError(
            "Definitive outer-test run is locked. Pass --confirm-definitive and set "
            f"{env_name}=1 only after user audit."
        )
