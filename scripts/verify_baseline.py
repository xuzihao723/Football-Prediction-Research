"""Check inherited source, data, and result bytes against the migration manifest."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "docs" / "provenance"


def sha256(path: Path) -> str:
    # The Europe archive already contained paths that exceed legacy Windows MAX_PATH.
    if os.name == "nt":
        path = Path("\\\\?\\" + os.path.abspath(path))
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-local", action="store_true", help="Require all local-only files.")
    parser.add_argument("--source-root", type=Path, help="Parent of the original source checkouts.")
    args = parser.parse_args()

    tracked = set(
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    )
    counts = Counter()
    errors = []
    overrides = json.loads((PROVENANCE / "optimization-overrides.json").read_text(encoding="utf-8"))
    with (PROVENANCE / "migration-manifest.tsv").open(encoding="utf-8", newline="") as file:
        for entry in csv.DictReader(file, delimiter="\t"):
            target = entry["target_path"]
            local = entry["placement"] == "local-only"
            if (target in tracked) == local:
                errors.append(f"Git index does not match placement of {target}")
            expected_target = entry["target_sha256"]
            override = overrides.pop(target, None)
            if override is not None:
                if local or not target.endswith(".py") or override["integration_sha256"] != expected_target:
                    errors.append(f"Invalid optimization override for {target}")
                else:
                    expected_target = override["optimized_sha256"]
            checks = []
            if not local or args.include_local:
                checks.append((ROOT / target, expected_target))
            if args.source_root is not None:
                source = args.source_root / entry["source"] / entry["source_path"]
                checks.append((source, entry["source_sha256"]))

            for path, expected in checks:
                try:
                    actual = sha256(path)
                except OSError as exc:
                    errors.append(f"{path}: {exc.strerror}")
                    continue
                if actual != expected:
                    errors.append(f"{path}: expected SHA-256 {expected}, found {actual}")
            counts[entry["placement"] if checks else "local-only excluded"] += 1

    errors.extend(f"Unknown optimization override: {target}" for target in overrides)
    for issue in errors:
        print(issue)
    summary = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
    print(f"Manifest entries ({summary}), discrepancies: {len(errors)}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
