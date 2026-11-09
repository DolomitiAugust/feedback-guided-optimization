"""Offline safeguards against combining unrelated runs or replacing evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

POLICY = Path(__file__).resolve().parents[1] / "policy_adaptation"
sys.path.insert(0, str(POLICY))
from collect_ppo_results import METRIC, main, validate_evaluations, write_results
from train_ppo_lora import normalize_pretrain


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.trained_path = self.root / "trained.json"
        self.config = {"pretrain": "Qwen/Qwen2.5-0.5B", "precision": "bf16"}
        self.before = {
            "model": self.config["pretrain"], "adapter": None,
            "precision": "torch.bfloat16", "decoding": "greedy, max_new_tokens=32",
            "metric": METRIC, "n": 128, "accuracy": 1.0,
            "results": [{"input": f"{n} + 1", "label": str(n + 1),
                         "response": str(n + 1), "score": 1.0} for n in range(128)],
        }
        self.after = copy.deepcopy(self.before)
        self.after["adapter"] = str(self.run / "actor_adapter")

    def validate(self):
        validate_evaluations(self.before, self.after, self.config, self.run, self.trained_path)

    def test_valid_absolute_and_legacy_relative_paths_without_fabricating_digest(self):
        original = copy.deepcopy(self.after)
        self.validate()
        self.assertEqual(self.after, original)
        self.after["adapter"] = "run/actor_adapter"
        self.validate()
        self.assertNotIn("actor_adapter_sha256", self.after)

    def test_different_base_model_rejected(self):
        self.after["model"] = "other/model"
        with self.assertRaisesRegex(ValueError, "different models"):
            self.validate()

    def test_training_model_mismatch_rejected(self):
        self.config["pretrain"] = "other/model"
        with self.assertRaisesRegex(ValueError, "training run"):
            self.validate()

    def test_baseline_adapter_rejected(self):
        self.before["adapter"] = self.after["adapter"]
        with self.assertRaisesRegex(ValueError, "without an adapter"):
            self.validate()

    def test_wrong_run_adapter_rejected(self):
        self.after["adapter"] = str(self.root / "another_run" / "actor_adapter")
        with self.assertRaisesRegex(ValueError, "does not belong"):
            self.validate()

    def test_same_questions_in_different_order_rejected(self):
        self.after["results"].reverse()
        with self.assertRaisesRegex(ValueError, "same ordered questions"):
            self.validate()

    def test_misgraded_response_rejected(self):
        self.after["results"][0]["response"] = "999"
        with self.assertRaisesRegex(ValueError, "scores disagree"):
            self.validate()

    def test_precision_mismatch_rejected(self):
        self.config["precision"] = "fp32"
        with self.assertRaisesRegex(ValueError, "precision"):
            self.validate()

    def test_existing_outputs_are_retained_without_partial_new_output(self):
        for filename in ("cloud_metrics.csv", "cloud_results.json"):
            with self.subTest(filename=filename):
                output = self.root / filename.replace(".", "_")
                output.mkdir()
                original = output / filename
                original.write_bytes(b"retain previous evidence")
                with self.assertRaises(FileExistsError):
                    write_results(output, [], {"training_executed": True})
                self.assertEqual(original.read_bytes(), b"retain previous evidence")
                self.assertEqual(list(output.iterdir()), [original])

    def test_validated_output_round_trip(self):
        output = self.root / "new_results"
        scalars = [{"tag": "train/reward", "step": 1, "value": 0.5, "wall_time": 123}]
        write_results(output, scalars, {"validation_n": 128})
        self.assertIn("train/reward,1,0.5,123", (output / "cloud_metrics.csv").read_text())
        self.assertEqual(json.loads((output / "cloud_results.json").read_text()), {"validation_n": 128})

    def test_invalid_pair_does_not_create_any_collection_output(self):
        self.after["adapter"] = str(self.root / "wrong" / "actor_adapter")
        baseline = self.root / "baseline.json"
        baseline.write_text(json.dumps(self.before), encoding="utf-8")
        self.trained_path.write_text(json.dumps(self.after), encoding="utf-8")
        (self.run / "run_config.json").write_text(json.dumps(self.config), encoding="utf-8")
        output = self.root / "should_not_exist"
        with patch.object(sys, "argv", ["collect_ppo_results.py", "--run-dir", str(self.run),
                         "--baseline", str(baseline), "--trained", str(self.trained_path),
                         "--output-dir", str(output)]):
            with self.assertRaisesRegex(ValueError, "does not belong"):
                main()
        self.assertFalse(output.exists())

    def test_published_legacy_records_remain_readable(self):
        results = POLICY / "results"
        before = json.loads((results / "baseline_validation.json").read_text(encoding="utf-8"))
        after = json.loads((results / "trained_validation.json").read_text(encoding="utf-8"))
        run = results / "ppo_outputs"
        config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        validate_evaluations(before, after, config, run, results / "trained_validation.json")


class LauncherPathTests(unittest.TestCase):
    def test_existing_relative_model_directory_becomes_absolute(self):
        relative = Path("policy_adaptation")
        self.assertTrue(relative.is_dir(), "Run tests from the repository root.")
        self.assertEqual(normalize_pretrain(str(relative)), str(relative.resolve()))

    def test_remote_model_id_is_unchanged(self):
        self.assertEqual(normalize_pretrain("Qwen/Qwen2.5-0.5B"), "Qwen/Qwen2.5-0.5B")


if __name__ == "__main__":
    unittest.main(verbosity=2)
