"""Generate a reproducible, disjoint arithmetic train/validation split."""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = (
    "Compute the sum. Write only the integer answer on the first line.\n"
    "Question: 2 + 3\nAnswer: 5\n"
    "Question: 7 + 1\nAnswer: 8\n"
)


def make_rows(seed=1234):
    # Canonical pairs prevent a+b and b+a from crossing the split.
    pairs = [(a, b) for a in range(50) for b in range(a, 50)
             if (a, b) not in {(2, 3), (1, 7)}]
    random.Random(seed).shuffle(pairs)
    rows = [{"input": PREFIX + f"Question: {a} + {b}\nAnswer: ",
             "label": str(a + b), "a": a, "b": b}
            for a, b in pairs[:640]]
    return rows[:512], rows[512:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(("train.jsonl", "validation.jsonl"), make_rows()):
        path = args.output_dir / name
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite different data: {path}")
        path.write_text(content, encoding="utf-8")
        print(f"{name}: {len(rows)} unique questions")


if __name__ == "__main__":
    main()
