"""Exercise the manual smoke entry with tiny local stubs, without model downloads."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy_adaptation"))
import lora_training as smoke


class LocalBatch(dict):
    def to(self, device):
        return self


class LocalTokenizer:
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, *args, **kwargs):
        return LocalBatch(input_ids=torch.ones(1, 2))


class LocalModel(nn.Module):
    def __init__(self, fail=False, nonfinite=False):
        super().__init__()
        self.q_proj = nn.Linear(2, 3)
        self.config = SimpleNamespace(pad_token_id=None)
        self.fail = fail
        self.nonfinite = nonfinite

    def forward(self, input_ids):
        if self.fail:
            raise RuntimeError("simulated forward failure")
        logits = self.q_proj(input_ids)
        if self.nonfinite:
            logits = logits * float("nan")
        return SimpleNamespace(logits=logits)


class SmokeEntryTests(unittest.TestCase):
    def run_entry(self, model):
        with patch.object(smoke, "select_device_dtype", return_value=("cpu", torch.float32)), \
             patch.object(smoke.AutoTokenizer, "from_pretrained", return_value=LocalTokenizer()) as tokenizer, \
             patch.object(smoke.AutoModelForCausalLM, "from_pretrained", return_value=model) as loader, \
             redirect_stdout(io.StringIO()):
            smoke.main()
        return tokenizer, loader

    def test_pinned_model_revision_success_and_merge(self):
        model = LocalModel()
        tokenizer, loader = self.run_entry(model)
        tokenizer.assert_called_once_with(smoke.MODEL, revision=smoke.MODEL_REVISION, use_fast=True)
        loader.assert_called_once_with(smoke.MODEL, revision=smoke.MODEL_REVISION, torch_dtype=torch.float32)
        self.assertTrue(model.q_proj.merged)
        self.assertFalse(model.training)
        self.assertFalse(model.q_proj.weight.requires_grad)

    def test_forward_failure_is_not_swallowed(self):
        with self.assertRaisesRegex(RuntimeError, "simulated forward failure"):
            self.run_entry(LocalModel(fail=True))

    def test_nonfinite_logits_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite logits"):
            self.run_entry(LocalModel(nonfinite=True))

    def test_dtype_selection_checks_native_bf16_support(self):
        for cuda, bf16, expected in ((False, False, ("cpu", torch.float32)),
                                     (True, False, ("cuda", torch.float32)),
                                     (True, True, ("cuda", torch.bfloat16))):
            with self.subTest(cuda=cuda, native_bf16=bf16):
                with patch.object(torch.cuda, "is_available", return_value=cuda), \
                     patch.object(torch.cuda, "is_bf16_supported", return_value=bf16) as capability:
                    self.assertEqual(smoke.select_device_dtype(), expected)
                    if cuda:
                        capability.assert_called_once_with(including_emulation=False)
                    else:
                        capability.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
