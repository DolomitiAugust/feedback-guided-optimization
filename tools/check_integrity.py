"""Verify released files against the content manifest using the standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ARTIFACTS_SHA256.json"


def files():
    if (ROOT / ".git").is_dir():
        names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
        for name in sorted(set(names.decode("utf-8").split("\0")) - {"", MANIFEST.name}):
            p = ROOT / name
            if p.is_file():
                yield p
        return
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p == MANIFEST:
            continue
        if any(x in p.relative_to(ROOT).parts for x in (".git", "__pycache__", ".venv", ".pytest_cache")):
            continue
        if p.name.endswith(("_grayscale.png", "_audit.json")) or p.name == "figure_validation.json":
            continue
        yield p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true", help="Create/update hashes after reviewing changes")
    args = parser.parse_args()
    if args.write_manifest:
        entries = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files()}
        MANIFEST.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Recorded {len(entries)} file hashes")
        return
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors = []
    for name, expected in entries.items():
        p = ROOT / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            errors.append(name)
    if errors:
        raise SystemExit("Missing or changed release files: " + ", ".join(errors))
    print(f"Verified {len(entries)} released files")


if __name__ == "__main__":
    main()
