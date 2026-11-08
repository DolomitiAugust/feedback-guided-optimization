"""Checked optional FlashAttention imports for the pinned single-GPU eager path.

This is a source compatibility patch, not a FlashAttention implementation.
Actual ring all_gather calls still import and call the original dependency.
OpenRLHF dependency metadata is retained; pip check can still report Flash absent.
Importing this module does not import OpenRLHF, start an installer or run training.
"""
import ast
import difflib
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import tempfile

COMMIT = "bfa85f6fa8cd6214707f93883fac5d08334f4f06"
REPOSITORY = "https://github.com/OpenRLHF/OpenRLHF"
SOURCE_HASHES = {
    "openrlhf/models/actor.py": "92e2adda7d2fa6459925f47c99072389ce71bde1e815ef08820d78e78d365baf",
    "openrlhf/models/model.py": "f239d25fb36939ec7b9b0247ff37d3a93f2f16f45f93f365b22bc489fc9fb7b3",
    "openrlhf/trainer/dpo_trainer.py": "baa3dc797736eea17b4a4329ba9ea6df417f0af91216f82e7107271ca7df3de8",
}
PATCHED_HASHES = {
    "openrlhf/models/actor.py": "bf0d57cdc5064f9edacbf49b236c764f4aff304304bf5a1edd1608bd0a97107d",
    "openrlhf/models/model.py": "9788f6f23d8e4d77d6f82e7343315b17828af8e5b99633910766e823c1b0258c",
    "openrlhf/trainer/dpo_trainer.py": "64c1a87a037165c464ee5aba270f503445c449a940463e4b04864454d07a1329",
}
OLD_IMPORT = "from flash_attn.utils.distributed import all_gather"
LAZY_WRAPPER = """def all_gather(*args, **kwargs):
    from flash_attn.utils.distributed import all_gather as _flash_all_gather
    return _flash_all_gather(*args, **kwargs)"""


def digest(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def validate_eager_configuration(argv, *, world_size=1):
    """Fail before launching any Flash, sample-packing or ring configuration."""
    if world_size != 1:
        raise RuntimeError("The optional-import path is restricted to one process/GPU.")
    forbidden = {"--flash_attn", "--packing_samples", "--ring_attn_size"}
    if any(str(argument).split("=", 1)[0] in forbidden for argument in argv):
        raise RuntimeError("Flash, packing and ring options are not allowed on this path.")


def _check_ast(original, patched):
    """Prove every original AST node except the one import is unchanged."""
    first, second = ast.parse(original), ast.parse(patched)
    imports = [node for node in first.body if isinstance(node, ast.ImportFrom)
               and node.module == "flash_attn.utils.distributed"]
    wrappers = [node for node in second.body if isinstance(node, ast.FunctionDef)
                and node.name == "all_gather"]
    expected_wrapper = ast.parse(LAZY_WRAPPER).body[0]
    if (len(imports) != 1 or len(wrappers) != 1
            or ast.dump(wrappers[0], include_attributes=False)
            != ast.dump(expected_wrapper, include_attributes=False)):
        raise RuntimeError("Unexpected optional-import AST.")
    first.body.remove(imports[0])
    second.body.remove(wrappers[0])
    if ast.dump(first, include_attributes=False) != ast.dump(second, include_attributes=False):
        raise RuntimeError("The optional-import patch changed other program logic.")


def checked_sources(relative, source):
    """Return (upstream, patched), accepting only exact upstream/patched hashes."""
    if relative not in SOURCE_HASHES:
        raise RuntimeError("File is outside the three-file patch allowlist.")
    current = digest(source)
    if current == SOURCE_HASHES[relative]:
        if source.count(OLD_IMPORT) != 1:
            raise RuntimeError("Expected exactly one upstream import.")
        original = source
        patched = source.replace(OLD_IMPORT, LAZY_WRAPPER, 1)
    elif current == PATCHED_HASHES[relative]:
        if source.count(LAZY_WRAPPER) != 1:
            raise RuntimeError("Expected exactly one checked lazy import.")
        patched = source
        original = source.replace(LAZY_WRAPPER, OLD_IMPORT, 1)
    else:
        raise RuntimeError("Unexpected source hash; refusing optional-import patch: " + relative)
    if digest(original) != SOURCE_HASHES[relative] or digest(patched) != PATCHED_HASHES[relative]:
        raise RuntimeError("Optional-import hash validation failed.")
    _check_ast(original, patched)
    return original, patched


def _check_origin(origin):
    info = origin.get("vcs_info", {})
    url = origin.get("url", "").rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    if (url.lower() != REPOSITORY.lower() or info.get("vcs") != "git"
            or info.get("commit_id") != COMMIT):
        raise RuntimeError("Install the exact upstream VCS commit; metadata will not be fabricated.")


def _atomic_write(path, content):
    """Replace one validated source while removing this helper's temporary file."""
    temporary = None
    mode = path.stat().st_mode
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".",
                                         suffix=".optional-flash.tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def apply_optional_flash_patch(log_dir):
    """Check all three files before writing any; save diffs and hash provenance.

    Returns changed_files and a stable provenance record. Repeated calls leave
    already patched sources untouched. No package metadata is changed.
    """
    installed = distribution("openrlhf")
    origin = json.loads(installed.read_text("direct_url.json") or "{}")
    _check_origin(origin)
    plans = []
    records = []
    for relative in SOURCE_HASHES:
        path = Path(installed.locate_file(relative))
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("Expected a regular installed source file: " + relative)
        raw = path.read_bytes()
        source = raw.decode("utf-8").replace("\r\n", "\n")
        original, patched = checked_sources(relative, source)
        plans.append((relative, path, raw, patched.encode("utf-8"), source != patched))
        records.append({"file": relative, "upstream_sha256": digest(original),
                        "patched_sha256": digest(patched),
                        "ast_otherwise_identical": True})
    # Every source and the VCS origin have passed before any source is replaced.
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for relative, path, raw, output, changed in plans:
        original, patched = checked_sources(relative, output.decode("utf-8"))
        patch = "".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                            fromfile="a/" + relative, tofile="b/" + relative))
        name = "optional_flash_" + relative.replace("/", "__") + ".patch"
        (log_dir / name).write_text(patch, encoding="utf-8")
    committed = []
    try:
        for relative, path, raw, output, changed in plans:
            if changed:
                _atomic_write(path, output)
                committed.append((path, raw))
    except BaseException:
        for path, raw in reversed(committed):
            _atomic_write(path, raw)
        raise
    record = {
        "commit": COMMIT, "repository": REPOSITORY, "patches": records,
        "purpose": "Delay unused ring all_gather imports for single-GPU eager unpacked PPO",
        "actual_ring_calls_still_require_flash": True,
        "package_metadata_unchanged": True,
        "pip_check_expected_exception": "OpenRLHF still declares flash-attn; it is intentionally absent on this restricted path",
        "gpu_training_validation": "not established by this source patch",
    }
    (log_dir / "optional_flash_patch.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {"changed_files": [relative for relative, _, _, _, changed in plans if changed],
            "record": record}

