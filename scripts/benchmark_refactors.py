"""Compare the integrated and optimized kernels with local research data."""

import argparse
import contextlib
import gc
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import types

from verify_baseline import sha256


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = "fcef30bae889974fd830fd48b5e17f30681355a9"
EUROPE = (
    ROOT
    / "projects/europe-big-five/FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction"
)
GULF = ROOT / "projects/gulf-market-anchored"


def load_module(path: Path, name: str, before: bool) -> tuple[types.ModuleType, str]:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    if before:
        relative = path.relative_to(ROOT).as_posix()
        source = subprocess.check_output(
            ["git", "show", f"{INTEGRATION}:{relative}"], cwd=ROOT
        )
    else:
        source = path.read_bytes()
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def measure(operation, repeats: int) -> dict:
    seconds = []
    hashes = set()
    shape = None
    for index in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fingerprint, shape = operation(index)
        seconds.append(time.perf_counter() - start)
        hashes.add(fingerprint())
    if len(hashes) != 1:
        raise AssertionError("Output fingerprint changed between repeats")
    return {
        "seconds": seconds,
        "median_without_warmup_seconds": statistics.median(seconds[1:]),
        "output_sha256": hashes.pop(),
        "output_shape": shape,
    }


def europe(repeats: int) -> dict:
    script = EUROPE / "build_momentum_features.py"
    data = EUROPE / "real_time_data/event_state_table_big5_2015_16.csv"
    result = {"input_sha256": sha256(data)}
    with tempfile.TemporaryDirectory(prefix="momentum-benchmark-") as directory:
        for variant in ("before", "after"):
            module, fingerprint = load_module(
                script, f"event_{variant}", before=variant == "before"
            )

            def run(index):
                output = Path(directory) / f"{variant}-{index}.csv"
                module.OUTPUT_CSV = output
                with contextlib.redirect_stdout(io.StringIO()):
                    features = module.build_momentum_features()
                return lambda: sha256(output), features.shape

            result[variant] = measure(run, repeats)
            result[variant]["code_sha256"] = fingerprint
    return result


def gulf(repeats: int) -> dict:
    import pandas as pd

    sys.path.insert(0, str(GULF))
    script = GULF / "src/pooled_split.py"
    data = GULF / "official_data/modeling/ModelingData_harmonized.csv"
    fixtures = pd.read_csv(data, low_memory=False)
    result = {"input_sha256": sha256(data)}
    for variant in ("before", "after"):
        module, code_hash = load_module(
            script, f"src.split_{variant}", before=variant == "before"
        )

        def run(_index):
            batches = [batch.tolist() for batch in module.temporal_batches(fixtures)]

            def fingerprint():
                payload = json.dumps(batches, separators=(",", ":")).encode()
                return hashlib.sha256(payload).hexdigest()

            return fingerprint, (len(fixtures), len(batches))

        result[variant] = measure(run, repeats)
        result[variant]["code_sha256"] = code_hash
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=("europe-event", "gulf"))
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two repetitions are required")
    benchmark = europe if args.project == "europe-event" else gulf
    result = benchmark(args.repeats)
    before, after = result["before"], result["after"]
    if (before["output_sha256"], before["output_shape"]) != (
        after["output_sha256"],
        after["output_shape"],
    ):
        raise SystemExit("Baseline and optimized outputs differ")
    result.update(
        project=args.project,
        integration_commit=INTEGRATION,
        current_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        python=sys.version,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
