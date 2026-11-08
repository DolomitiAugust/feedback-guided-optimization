"""Check saved geometry and online/local provenance without rerunning search."""
import csv
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / "program_search"

def load(name):
    return json.loads((ROOT / "evidence" / name).read_text(encoding="utf-8"))

class SavedSearchTests(unittest.TestCase):
    def check_geometry(self, packing, expected):
        c, r = packing["centers"], packing["radii"]
        self.assertEqual((len(c), len(r)), (26, 26))
        self.assertTrue(all(len(p) == 2 for p in c))
        self.assertTrue(all(math.isfinite(x) for p in c for x in p))
        self.assertTrue(all(math.isfinite(x) and x >= 0 for x in r))
        self.assertAlmostEqual(sum(r), expected, places=13)
        for (x, y), rad in zip(c, r):
            self.assertGreaterEqual(min(x, y, 1-x, 1-y)-rad, -1e-10)
        for i in range(26):
            for j in range(i):
                self.assertGreaterEqual(math.dist(c[i], c[j])-r[i]-r[j], -1e-10)

    def test_selected_local_geometry(self):
        self.check_geometry(load("best_local_packing.json"), 2.6206092130170973)

    def test_initial_local_geometry(self):
        self.check_geometry(load("local_packing.json"), 0.9597642169962063)

    def test_online_and_local_are_distinct(self):
        online, local = load("online_result_summary.json"), load("verified_4705b8e92e1b.json")
        self.assertEqual(online["best_online_score"], 2.62001380699489)
        self.assertEqual(local["local_score"], 2.6206092130170973)
        self.assertNotEqual(local["online_score"], local["local_score"])
        self.assertEqual(online["best_candidate_id"], local["candidate_id"])
        self.assertEqual(online["valid_evaluated_candidates"], 5)
        self.assertEqual(online["saved_candidate_count"], 7)

    def test_online_sequence(self):
        with (ROOT / "evidence/online_evaluations.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 5)
        best = max(rows, key=lambda row: float(row["score"]))
        self.assertEqual(best["candidate_id"], "4705b8e92e1b")
        self.assertEqual(int(best["generation"]), 0)
        self.assertEqual(max(float(x["score"]) for x in rows), 2.62001380699489)

if __name__ == "__main__":
    unittest.main(verbosity=2)
