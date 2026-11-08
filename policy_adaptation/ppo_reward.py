"""Verifiable arithmetic rewards for OpenRLHF v0.6.0.

The launcher uses the HTTP service in ppo_reward_server.py. The reward_func
wrapper is also available for local tensor-based checks. Each query contains
the prompt AND the generated response, including possible Qwen padding/EOS
tokens. Only the response is scored; prompt numbers cannot earn a reward.
"""
import re

SPECIAL_TOKENS = ("<|endoftext|>", "<|im_end|>")
ANSWER_PATTERN = re.compile(r"[ \t]*([+-]?[0-9]+)[ \t]*")


def response_only(query: str, prompt: str) -> str:
    for token in SPECIAL_TOKENS:
        query = query.replace(token, "")
        prompt = prompt.replace(token, "")
    if not query.startswith(prompt):
        # Reject a mismatched or truncated prompt instead of grading its text.
        return ""
    return query[len(prompt):]


def score_response(response: str, label: str) -> float:
    # A base language model may continue the few-shot examples. Grade only
    # its first answer line; never search later lines for the target number.
    first_line = response.split("\n", 1)[0].rstrip("\r")
    match = ANSWER_PATTERN.fullmatch(first_line)
    if match is None:
        return 0.0
    try:
        return float(int(match.group(1)) == int(label))
    except (TypeError, ValueError):
        return 0.0


def reward_func(queries, prompts, labels):
    import torch
    if not (len(queries) == len(prompts) == len(labels)):
        raise ValueError("queries, prompts and labels must have equal lengths")
    return torch.tensor(
        [score_response(response_only(q, p), str(y))
         for q, p, y in zip(queries, prompts, labels)],
        dtype=torch.float32,
    )
