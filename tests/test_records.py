"""Independent checks of the published observations; no external packages."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class RecordedEvidenceTests(unittest.TestCase):
    def test_matched_validation_from_responses(self):
        old = load("policy_adaptation/results/baseline_validation.json")
        new = load("policy_adaptation/results/trained_validation.json")
        for key in ("model", "precision", "metric", "decoding", "n"):
            self.assertEqual(old[key], new[key])
        counts = [0, 0, 0, 0]
        for a, b in zip(old["results"], new["results"]):
            for key in ("input", "a", "b", "label"):
                self.assertEqual(a[key], b[key])
            scores = []
            for r in (a, b):
                m = re.fullmatch(r"[ \t]*([+-]?[0-9]+)[ \t]*", r["response"].split("\n", 1)[0].rstrip("\r"))
                s = int(m is not None and int(m.group(1)) == int(r["label"]))
                self.assertEqual(s, r["score"])
                self.assertEqual(int(r["label"]), r["a"] + r["b"])
                scores.append(s)
            counts[2 * scores[0] + scores[1]] += 1
        self.assertEqual(len(old["results"]), 128)
        self.assertEqual(len(new["results"]), 128)
        self.assertEqual(counts, [6, 5, 0, 117])
        self.assertEqual(sum(r["score"] for r in old["results"]), 117)
        self.assertEqual(sum(r["score"] for r in new["results"]), 122)

    def test_full_metric_coverage_and_failed_comparison(self):
        with (ROOT / "policy_adaptation/results/cloud_metrics.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        tags = {}
        for r in rows:
            self.assertTrue(math.isfinite(float(r["value"])))
            tags.setdefault(r["tag"], []).append(int(r["step"]))
        self.assertEqual(len(rows), 320)
        self.assertEqual(len(tags), 10)
        for steps in tags.values():
            self.assertEqual(sorted(steps), list(range(1, 33)))
        success = load("policy_adaptation/results/cloud_results.json")
        self.assertEqual(success["training_exit_code"], 0)
        historical = load("policy_adaptation/logs/observed_attempt/summary.json")
        self.assertEqual(historical["training_exit_code"], 1)
        self.assertFalse(historical["post_training_evaluation_available"])
        self.assertEqual(historical["baseline_correct"], 118)

    def test_online_scores_are_not_local_scores(self):
        s = load("program_search/evidence/online_result_summary.json")
        self.assertEqual(s["saved_candidate_count"], 7)
        self.assertEqual(s["valid_evaluated_candidates"], 5)
        winner = max(s["rows"], key=lambda x: x["score"])
        self.assertEqual(winner["generation"], 0)
        self.assertEqual(winner["score"], 2.62001380699489)
        self.assertEqual(s["local_verification"]["local_score"], 2.6206092130170973)
        self.assertNotEqual(winner["score"], s["local_verification"]["local_score"])

    def test_saved_circle_geometry(self):
        p = load("program_search/evidence/best_local_packing.json")
        centers, radii = p["centers"], p["radii"]
        self.assertEqual(len(centers), 26)
        self.assertEqual(len(radii), 26)
        for i, ((x, y), r) in enumerate(zip(centers, radii)):
            self.assertTrue(all(math.isfinite(v) for v in (x, y, r)))
            self.assertGreaterEqual(r, 0)
            self.assertGreaterEqual(min(x-r, y-r, 1-x-r, 1-y-r), -1e-10)
            for j in range(i):
                gap = math.hypot(x-centers[j][0], y-centers[j][1]) - r - radii[j]
                self.assertGreaterEqual(gap, -1e-10)
        self.assertAlmostEqual(sum(radii), 2.6206092130170973, places=12)

    def test_actor_weights_match_measured_record(self):
        s = load("policy_adaptation/results/cloud_results.json")
        p = ROOT / "policy_adaptation/results/ppo_outputs/actor_adapter/adapter_model.safetensors"
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), s["actor_adapter_sha256"])


if __name__ == "__main__":
    unittest.main()
