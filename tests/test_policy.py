"""Behavior tests: numerical equivalence, real updates, injection and rewards."""
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "policy_adaptation"))
from pathlib import Path
import unittest
import urllib.request
import urllib.error

import torch
from torch import nn

from lora_training import (LoRALinear, count_model_params, inject_lora_modules,
                           mark_only_lora_as_trainable)
from ppo_reward import response_only, reward_func, score_response
from prepare_ppo_data import make_rows
from ppo_reward_server import LocalRewardServer


class LoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_initial_equivalence_and_shared_parameters(self):
        base = nn.Linear(7, 5, dtype=torch.float64)
        layer = LoRALinear(base, r=3, alpha=6, dropout=0.1)
        x = torch.randn(2, 4, 7, dtype=torch.float64)
        self.assertIs(layer.weight, base.weight)
        self.assertIs(layer.bias, base.bias)
        self.assertEqual(layer.lora_A.weight.dtype, base.weight.dtype)
        torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)

    def test_nonzero_adapter_forward_and_merge(self):
        layer = LoRALinear(nn.Linear(7, 5, dtype=torch.float64), r=3, alpha=6, dropout=0.2)
        with torch.no_grad():
            layer.lora_B.weight.normal_()
        x = torch.randn(2, 4, 7, dtype=torch.float64)
        layer.eval()
        expected_weight = layer.weight.detach().clone() + 2 * (layer.lora_B.weight @ layer.lora_A.weight)
        expected = nn.functional.linear(x, expected_weight, layer.bias)
        before = layer(x).detach()
        torch.testing.assert_close(before, expected, rtol=1e-12, atol=1e-12)
        layer.merge()
        torch.testing.assert_close(layer(x), before, rtol=1e-12, atol=1e-12)
        weight = layer.weight.detach().clone()
        layer.merge()
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
        with torch.no_grad():
            layer.lora_B.weight.add_(10)
        torch.testing.assert_close(layer(x), before, rtol=1e-12, atol=1e-12)

    def test_only_adapter_updates_after_training(self):
        layer = LoRALinear(nn.Linear(7, 5), r=3, alpha=6)
        expected_count = 3 * (7 + 5)
        self.assertEqual(mark_only_lora_as_trainable(layer), expected_count)
        base = layer.weight.detach().clone()
        bias = layer.bias.detach().clone()
        a = layer.lora_A.weight.detach().clone()
        b = layer.lora_B.weight.detach().clone()
        optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.05)
        x, target = torch.randn(8, 7), torch.randn(8, 5)
        for _ in range(3):
            optimizer.zero_grad()
            nn.functional.mse_loss(layer(x), target).backward()
            optimizer.step()
        torch.testing.assert_close(layer.weight, base, rtol=0, atol=0)
        torch.testing.assert_close(layer.bias, bias, rtol=0, atol=0)
        self.assertIsNone(layer.weight.grad)
        self.assertFalse(torch.equal(layer.lora_A.weight, a))
        self.assertFalse(torch.equal(layer.lora_B.weight, b))

    def test_nested_and_top_level_injection(self):
        model = nn.Module()
        model.q_proj = nn.Linear(7, 5)
        model.block = nn.Module()
        model.block.down_proj = nn.Linear(5, 3)
        model.other = nn.Linear(5, 3)
        original_weight = model.block.down_proj.weight
        model.eval()
        self.assertEqual(inject_lora_modules(model, r=2), 2)
        self.assertIs(model.block.down_proj.weight, original_weight)
        self.assertFalse(model.q_proj.training)
        self.assertIsInstance(model.other, nn.Linear)
        self.assertEqual(inject_lora_modules(model, r=2), 0)
        mark_only_lora_as_trainable(model)
        trainable, total, ratio = count_model_params(model)
        self.assertEqual(trainable, 2 * (7 + 5) + 2 * (5 + 3))
        self.assertEqual(total, (35 + 5) + (15 + 3) * 2 + trainable)
        self.assertAlmostEqual(ratio, trainable / total)

    def test_zero_rank_and_empty_model(self):
        base = nn.Linear(3, 2, bias=False)
        layer = LoRALinear(base, r=0)
        x = torch.randn(2, 3)
        torch.testing.assert_close(layer(x), base(x))
        layer.merge()
        torch.testing.assert_close(layer(x), base(x))
        self.assertEqual(count_model_params(nn.Module()), (0, 0, 0.0))

    def test_invalid_configuration(self):
        with self.assertRaises(TypeError):
            LoRALinear(nn.ReLU())
        for rank in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                LoRALinear(nn.Linear(3, 2), r=rank)
        for dropout in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                LoRALinear(nn.Linear(3, 2), dropout=dropout)


class PPOContractTests(unittest.TestCase):
    def test_reward_grades_response_only(self):
        prompt = "Question: 2 + 3\nAnswer: "
        rewards = reward_func(
            [prompt + "5<|endoftext|>",
             prompt + "7", prompt, "unrelated 5"],
            [prompt] * 4, ["5"] * 4)
        self.assertEqual(rewards.tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(response_only("<|endoftext|>" + prompt + "5", prompt), "5")
        self.assertEqual(score_response("5 extra text", "5"), 0.0)
        self.assertEqual(score_response("4\n5", "5"), 0.0)
        self.assertEqual(score_response("5\nQuestion: 2 + 2\nAnswer: 4", "5"), 1.0)
        self.assertEqual(score_response("5.0", "5"), 0.0)
        self.assertEqual(score_response("\n5", "5"), 0.0)
        self.assertEqual(score_response(" -5 \r\n", "-5"), 1.0)
        self.assertEqual(score_response("<answer>5</answer>", "5"), 0.0)
        with self.assertRaises(ValueError):
            reward_func(["a"], [], [])

    def test_data_labels_and_split(self):
        train, validation = make_rows()
        self.assertEqual((len(train), len(validation)), (512, 128))
        self.assertEqual(make_rows(), (train, validation))
        left = {(r["a"], r["b"]) for r in train}
        right = {(r["a"], r["b"]) for r in validation}
        self.assertEqual(len(left), 512)
        self.assertEqual(len(right), 128)
        self.assertFalse(left & right)
        for row in train + validation:
            self.assertEqual(int(row["label"]), row["a"] + row["b"])

    def test_real_reward_http_contract(self):
        prompt = "Question: 2 + 3\nAnswer: "
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with LocalRewardServer() as server:
            payload = {"query": [prompt + "5", prompt + "8"],
                       "prompts": [prompt, prompt], "labels": ["5", "5"]}
            request = urllib.request.Request(server.url, data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            with opener.open(request, timeout=5) as response:
                self.assertEqual(json.load(response), {"rewards": [1.0, 0.0]})
            payload["labels"] = []
            request = urllib.request.Request(server.url, data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                opener.open(request, timeout=5)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
