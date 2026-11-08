"""Apply checked compatibility fixes to the pinned OpenRLHF v0.6.0 source.

The pinned upstream constructor requires pad_len, but its single-GPU caller
omits it. Passing None matches the upstream unpacked Ray path. No PPO math,
sampling policy, optimizer, or training budget is changed by these patches.
The save fix recognizes a PEFT-prefixed language-model head only when it is
the exact same Parameter as the input embeddings, retaining the key check.
"""
import ast
from dataclasses import dataclass
from contextlib import nullcontext
import difflib
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Union

COMMIT = "bfa85f6fa8cd6214707f93883fac5d08334f4f06"
SOURCE_SHA256 = "19426ae1106575b7080b02e80ae754d466244ad0b19c175c06b060e2fdb8666f"
RELATIVE_SOURCE = "openrlhf/trainer/ppo_utils/experience_maker.py"
OLD = "                labels=labels,\n            )\n            samples_list.append(samples)"
NEW = "                labels=labels,\n                pad_len=None,\n            )\n            samples_list.append(samples)"
ROOT = Path(__file__).resolve().parent
SAVE_SOURCE = "openrlhf/utils/deepspeed/deepspeed.py"
SAVE_SHA256 = "426198b6feafb853c1fb9f8442ef7e32250b51aff77a353203dc4d341848aabe"
SAVE_OLD = '''            if getattr(model_to_save.config, "tie_word_embeddings", False) and "lm_head.weight" in state_dict_keys:
                state_dict_keys.remove("lm_head.weight")'''
SAVE_NEW = '''            if getattr(model_to_save.config, "tie_word_embeddings", False):
                for key in list(state_dict_keys - output_state_dict_keys):
                    if key.endswith("lm_head.weight") and (
                        model_to_save.get_parameter(key) is model_to_save.get_input_embeddings().weight
                    ):
                        state_dict_keys.remove(key)'''


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def patched_source(source):
    if NEW in source and digest(source.replace(NEW, OLD, 1)) == SOURCE_SHA256:
        return source
    if digest(source) != SOURCE_SHA256 or source.count(OLD) != 1:
        raise RuntimeError("Unexpected OpenRLHF source; refusing an unchecked patch.")
    return source.replace(OLD, NEW, 1)


def patched_save_source(source):
    if SAVE_NEW in source and digest(source.replace(SAVE_NEW, SAVE_OLD, 1)) == SAVE_SHA256:
        return source
    if digest(source) != SAVE_SHA256 or source.count(SAVE_OLD) != 1:
        raise RuntimeError("Unexpected OpenRLHF save source; refusing an unchecked patch.")
    return source.replace(SAVE_OLD, SAVE_NEW, 1)


def verify_sampler(source):
    """Run the actual upstream sampler method with tiny CPU test doubles."""
    import torch
    tree = ast.parse(source)
    samples = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "Samples")
    maker = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "NaiveExperienceMaker")
    generate = next(node for node in maker.body if isinstance(node, ast.FunctionDef)
                    and node.name == "generate_samples")
    namespace = {"torch": torch, "dataclass": dataclass, "Optional": Optional,
                 "Union": Union, "List": List}
    exec(compile(ast.Module(body=[samples, generate], type_ignores=[]),
                 "<upstream sampler regression check>", "exec"), namespace)
    actor = SimpleNamespace(
        eval=lambda: None,
        generate=lambda **kwargs: (torch.tensor([[1, 2, 3]]),
                                  torch.ones(1, 3, dtype=torch.long),
                                  torch.ones(1, 1, dtype=torch.bool)),
    )
    fake = SimpleNamespace(
        actor=actor, prompt_max_len=192,
        strategy=SimpleNamespace(args=SimpleNamespace(
            n_samples_per_prompt=1, micro_rollout_batch_size=1)),
        tokenize_fn=lambda prompts, max_length, device: {},
    )
    result = namespace["generate_samples"](fake, ["test prompt"], ["3"])
    assert len(result) == 1 and result[0].pad_len is None
    assert result[0].prompts == ["test prompt"] and result[0].labels == ["3"]
    assert result[0].num_actions == 1


def verify_save_method(source, *, tied=True):
    """Exercise the actual save method on a tiny model with PEFT-style names.

The untied case deliberately omits an independent head from the gathered
parameters: it must still fail, proving the assertion has not been bypassed.
    """
    import torch
    from torch import nn
    tree = ast.parse(source)
    strategy = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "DeepspeedStrategy")
    save = next(node for node in strategy.body if isinstance(node, ast.FunctionDef)
                and node.name == "save_model")
    saved = {}
    class WrappedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = nn.Module()
            self.base_model.model = nn.Module()
            self.base_model.model.embed_tokens = nn.Embedding(5, 3)
            self.base_model.model.lm_head = nn.Linear(3, 5, bias=False)
            if tied:
                self.base_model.model.lm_head.weight = self.base_model.model.embed_tokens.weight
            self.config = SimpleNamespace(tie_word_embeddings=True, _name_or_path="no-source",
                                          to_json_file=lambda path: saved.update(config=True))
        def named_parameters(self, *args, **kwargs):
            for name, parameter in super().named_parameters(*args, **kwargs):
                if not tied and name.endswith("lm_head.weight"):
                    continue
                yield name, parameter
        def get_input_embeddings(self):
            return self.base_model.model.embed_tokens
        def save_pretrained(self, output_dir, **kwargs):
            saved["state_dict"] = kwargs["state_dict"]
    namespace = {
        "nn": nn, "torch": torch,
        "os": SimpleNamespace(makedirs=lambda *a, **k: None,
                              path=SimpleNamespace(join=os.path.join, exists=lambda path: False)),
        "deepspeed": SimpleNamespace(zero=SimpleNamespace(
            GatheredParameters=lambda *a, **k: nullcontext())),
        "_z3_params_to_fetch": lambda parameters: [],
        "PeftModel": type("UnusedPeftModel", (), {}),
    }
    exec(compile(ast.Module(body=[save], type_ignores=[]),
                 "<upstream save regression check>", "exec"), namespace)
    fake_strategy = SimpleNamespace(is_rank_0=lambda: True, _unwrap_model=lambda model: model, stage=2)
    tokenizer = SimpleNamespace(save_pretrained=lambda path: saved.update(tokenizer=True))
    namespace["save_model"](fake_strategy, WrappedModel(), tokenizer, "unused-output")
    assert saved.get("config") and saved.get("tokenizer") and saved.get("state_dict")


def apply_compatibility_patch():
    installed = distribution("openrlhf")
    origin = json.loads(installed.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != COMMIT:
        raise RuntimeError("The compatibility patch requires the pinned OpenRLHF commit.")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    records = []
    for relative, transform, old, new, verify, label in (
        (RELATIVE_SOURCE, patched_source, OLD, NEW, verify_sampler, "pad_len"),
        (SAVE_SOURCE, patched_save_source, SAVE_OLD, SAVE_NEW, verify_save_method, "save_model"),
    ):
        path = Path(installed.locate_file(relative))
        source = path.read_text(encoding="utf-8")
        patched = transform(source)
        verify(patched)
        if label == "save_model":
            try:
                verify_save_method(patched, tied=False)
            except AssertionError as error:
                assert "mismatch keys" in str(error)
            else:
                raise AssertionError("A missing independent head must not pass the save check.")
        original = patched.replace(new, old, 1)
        patch = "".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                           fromfile="a/" + relative, tofile="b/" + relative))
        (logs / f"openrlhf_{label}.patch").write_text(patch, encoding="utf-8")
        if patched != source:
            path.write_text(patched, encoding="utf-8")
        records.append({"file": relative, "upstream_sha256": digest(original),
                        "patched_sha256": digest(patched), "regression_check_passed": True})
    record = {"commit": COMMIT, "patches": records,
              "independent_missing_weight_still_rejected": True}
    (logs / "openrlhf_patch.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("OpenRLHF sampler and tied-weight save patches verified.", flush=True)
    return record


if __name__ == "__main__":
    apply_compatibility_patch()
