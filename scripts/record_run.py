"""Record an experiment's inputs, environment, metrics, and outputs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4

from verify_baseline import sha256


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {"europe-match", "europe-event", "gulf"}
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
SPEC_FIELDS = {
    "project",
    "experiment_id",
    "inputs",
    "configuration",
    "random_seed",
    "model_parameters",
    "training_protocol",
    "evaluation_metrics",
    "command",
}


def read_json(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(record, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return record


def write_new_json(path: Path, record: dict) -> None:
    serialized = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")


def describe_file(path: Path) -> dict:
    return {"size_bytes": path.stat().st_size, "sha256": sha256(path)}


def validate_spec(spec: dict) -> None:
    if missing := SPEC_FIELDS.difference(spec):
        raise ValueError(f"Missing experiment fields: {', '.join(sorted(missing))}")
    if spec["project"] not in PROJECTS:
        raise ValueError(f"Project must be one of {', '.join(sorted(PROJECTS))}")
    if not ID_PATTERN.fullmatch(spec["experiment_id"]):
        raise ValueError("experiment_id must be a safe name of 1–64 characters")
    for field in ("inputs", "configuration", "model_parameters", "training_protocol"):
        if not isinstance(spec[field], dict):
            raise ValueError(f"{field} must be a JSON object")
    if not all(isinstance(value, str) and value for value in spec["inputs"].values()):
        raise ValueError("Each input must name a file")
    for field in ("evaluation_metrics", "command"):
        if not isinstance(spec[field], list) or not all(
            isinstance(value, str) and value for value in spec[field]
        ):
            raise ValueError(f"{field} must be an array of strings")
    if not spec["command"]:
        raise ValueError("command must describe the intended invocation")
    if spec["random_seed"] is not None and type(spec["random_seed"]) is not int:
        raise ValueError("random_seed must be an integer or null")


def git_state() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
    changes = subprocess.check_output(
        [
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        cwd=ROOT,
    )
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"], cwd=ROOT)
    return {
        "commit": commit.strip(),
        "dirty": bool(changes),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def python_environment(python: Path) -> dict:
    code = """
import importlib.metadata, json, platform, sys
print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "packages": sorted({(d.metadata["Name"], d.version) for d in importlib.metadata.distributions()}),
}))
"""
    result = subprocess.check_output(
        [str(python), "-X", "utf8", "-c", code], encoding="utf-8"
    )
    environment = json.loads(result)
    environment["settings"] = {
        name: os.environ[name]
        for name in (
            "PYTHONHASHSEED",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
        if name in os.environ
    }
    return environment


def start(spec_path: Path, runs_dir: Path, python: Path) -> Path:
    spec = read_json(spec_path)
    validate_spec(spec)
    inputs = {}
    for key, name in spec["inputs"].items():
        path = Path(os.path.abspath(spec_path.parent / name))
        resolved = path.resolve(strict=True)
        inputs[key] = {
            "path": str(path),
            "resolved_path": str(resolved),
            **describe_file(resolved),
        }

    record = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "spec": spec,
        "inputs": inputs,
        "git": git_state(),
        "environment": python_environment(python),
    }
    run_id = f"{spec['project']}-{spec['experiment_id']}-{uuid4().hex[:12]}"
    run = runs_dir.resolve() / run_id
    (run / "outputs").mkdir(parents=True, exist_ok=False)
    record["run_id"] = run_id
    write_new_json(run / "start.json", record)
    return run


def finish(run: Path, exit_code: int, metrics_json: Path | None) -> bool:
    manifest = run / "finish.json"
    if manifest.exists():
        raise FileExistsError(f"Run already finalized: {run}")
    start_record = read_json(run / "start.json")
    outputs_dir = (run / "outputs").resolve()
    if not outputs_dir.is_dir() or outputs_dir.parent != run.resolve():
        raise ValueError("Expected a local outputs directory inside the run")
    outputs = {}
    for path in sorted(outputs_dir.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(outputs_dir):
            raise ValueError(f"Output must remain inside the run directory: {path}")
        if path.is_file():
            outputs[path.relative_to(outputs_dir).as_posix()] = describe_file(path)

    metrics = None
    if metrics_json is not None:
        path = (outputs_dir / metrics_json).resolve()
        if (
            not path.is_relative_to(outputs_dir)
            or path.relative_to(outputs_dir).as_posix() not in outputs
        ):
            raise ValueError("metrics JSON must be an output file within this run")
        metrics = read_json(path)

    input_drift = []
    for key, entry in start_record["inputs"].items():
        try:
            resolved = Path(entry["path"]).resolve(strict=True)
            same = (
                str(resolved) == entry["resolved_path"]
                and sha256(resolved) == entry["sha256"]
            )
        except OSError:
            same = False
        if not same:
            input_drift.append(key)
    status = "failed" if exit_code else "succeeded"
    end_git = git_state()
    code_drift = start_record["git"] != end_git
    if input_drift or code_drift:
        status = "invalid"
    write_new_json(
        manifest,
        {
            "schema_version": 1,
            "run_id": start_record["run_id"],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "exit_code": exit_code,
            "input_drift": input_drift,
            "code_drift": code_drift,
            "git": end_git,
            "code_provenance": "unverified"
            if code_drift or end_git["dirty"]
            else "verified",
            "metrics": metrics,
            "outputs": outputs,
        },
    )
    return status != "invalid"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    launch = sub.add_parser(
        "start", help="Fingerprint inputs and reserve a unique output directory"
    )
    launch.add_argument("--spec", required=True, type=Path)
    launch.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    launch.add_argument("--python", type=Path, default=Path(sys.executable))
    complete = sub.add_parser(
        "finish", help="Record the exit code and fingerprint outputs"
    )
    complete.add_argument("run", type=Path)
    complete.add_argument("--exit-code", required=True, type=int)
    complete.add_argument(
        "--metrics-json", type=Path, help="JSON output path relative to outputs/"
    )
    args = parser.parse_args()
    try:
        if args.action == "start":
            print(start(args.spec, args.runs_dir, args.python))
        elif not finish(args.run, args.exit_code, args.metrics_json):
            parser.exit(1, "Inputs or the Git state changed. Run marked invalid.\n")
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
