import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_run.py"


class RecordRunTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.script = SCRIPT
        self.data = self.directory / "data.txt"
        self.data.write_text("frozen input\n", encoding="utf-8")
        self.spec = self.directory / "spec.json"
        self.spec.write_text(
            json.dumps(
                {
                    "project": "europe-event",
                    "experiment_id": "first-half",
                    "inputs": {"events": "data.txt"},
                    "configuration": {"grid_size": 91},
                    "random_seed": None,
                    "model_parameters": {},
                    "training_protocol": {"type": "feature-only"},
                    "evaluation_metrics": ["checksum"],
                    "command": ["python", "build_momentum_features.py"],
                }
            ),
            encoding="utf-8",
        )

    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, str(self.script), *map(str, args)],
            capture_output=True,
            encoding="utf-8",
        )

    def start(self):
        launched = self.invoke(
            "start", "--spec", self.spec, "--runs-dir", self.directory / "runs"
        )
        self.assertEqual(launched.returncode, 0, launched.stderr)
        return Path(launched.stdout.strip())

    def test_records_input_environment_config_metrics_and_outputs(self):
        run = self.start()
        before = json.loads((run / "start.json").read_text(encoding="utf-8"))
        self.assertEqual(
            before["inputs"]["events"]["sha256"],
            hashlib.sha256(self.data.read_bytes()).hexdigest(),
        )
        self.assertEqual(before["spec"]["configuration"], {"grid_size": 91})
        self.assertEqual(len(before["git"]["commit"]), 40)
        self.assertIn("python", before["environment"])
        metrics = run / "outputs" / "metrics.json"
        metrics.write_text('{"checksum": "abc"}', encoding="utf-8")
        finished = self.invoke(
            "finish", run, "--exit-code", 0, "--metrics-json", "metrics.json"
        )
        self.assertEqual(finished.returncode, 0, finished.stderr)
        final = json.loads((run / "finish.json").read_text(encoding="utf-8"))
        self.assertEqual(final["status"], "succeeded")
        self.assertEqual(final["metrics"], {"checksum": "abc"})
        self.assertEqual(set(final["outputs"]), {"metrics.json"})
        self.assertNotEqual(self.invoke("finish", run, "--exit-code", 0).returncode, 0)

    def test_input_drift_invalidates_success(self):
        run = self.start()
        self.data.write_text("changed\n", encoding="utf-8")
        finished = self.invoke("finish", run, "--exit-code", 0)
        self.assertNotEqual(finished.returncode, 0)
        final = json.loads((run / "finish.json").read_text(encoding="utf-8"))
        self.assertEqual(final["status"], "invalid")
        self.assertEqual(final["input_drift"], ["events"])

    def test_failed_experiment_is_recorded(self):
        run = self.start()
        self.assertEqual(self.invoke("finish", run, "--exit-code", 7).returncode, 0)
        final = json.loads((run / "finish.json").read_text(encoding="utf-8"))
        self.assertEqual((final["status"], final["exit_code"]), ("failed", 7))

    def test_tracked_code_change_invalidates_provenance(self):
        repository = self.directory / "repository"
        scripts = repository / "scripts"
        scripts.mkdir(parents=True)
        self.script = scripts / SCRIPT.name
        shutil.copyfile(SCRIPT, self.script)
        shutil.copyfile(
            SCRIPT.with_name("verify_baseline.py"), scripts / "verify_baseline.py"
        )
        (repository / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
        tracked = repository / "model.py"
        tracked.write_text("setting = 1\n", encoding="utf-8")
        for command in (
            ["init", "-q"],
            ["add", "."],
            [
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@localhost",
                "commit",
                "-qm",
                "Test fixture",
            ],
        ):
            subprocess.run(
                ["git", *command], cwd=repository, check=True, capture_output=True
            )
        run = self.start()
        tracked.write_text("setting = 2\n", encoding="utf-8")
        self.assertNotEqual(self.invoke("finish", run, "--exit-code", 0).returncode, 0)
        final = json.loads((run / "finish.json").read_text(encoding="utf-8"))
        self.assertEqual(final["status"], "invalid")
        self.assertTrue(final["code_drift"])
