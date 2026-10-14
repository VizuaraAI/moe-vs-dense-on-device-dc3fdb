"""A small quality harness.

Purpose: make sure no arm of a comparison wins by being worse. Every latency
number in the paper must be reportable next to a quality number measured at the
same quantization and the same runtime.

This is deliberately small. It is a guard rail, not a leaderboard submission.
For headline quality numbers in the final paper, run a standard harness such as
lm-evaluation-harness and cite it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .registry import ModelSpec


@dataclass
class QualityResult:
    model_key: str
    task: str
    accuracy: float
    n_examples: int
    quantization: str
    note: str = ""


def _normalized_loglikelihood(model, tokenizer, context: str, continuation: str, device) -> float:
    """Mean log-probability per token of ``continuation`` given ``context``.

    Length-normalized, which is the standard choice for multiple-choice tasks
    where the options differ in length. Without normalization the shortest
    option wins too often.
    """
    import torch

    ctx_ids = tokenizer(context, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(context + continuation, return_tensors="pt")["input_ids"]
    cont_len = full_ids.shape[1] - ctx_ids.shape[1]
    if cont_len <= 0:
        return float("-inf")

    full_ids = full_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids=full_ids).logits

    logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = full_ids[:, 1:]
    gathered = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return float(gathered[0, -cont_len:].mean().item())


def evaluate_hellaswag(
    spec: ModelSpec,
    n_examples: int = 200,
    quantization: str = "none",
    split: str = "validation",
) -> QualityResult:
    """Length-normalized accuracy on a HellaSwag subset.

    Downloads the public dataset on first use. 200 examples is enough to catch
    a model that has been broken by quantization, which is what this guard is
    for; it is not enough to rank two healthy models.
    """
    from datasets import load_dataset

    from .backends import HFBackend

    backend = HFBackend(spec, quantization=quantization)
    backend.load()
    model, tokenizer = backend.model, backend.tokenizer
    device = model.device

    dataset = load_dataset("Rowan/hellaswag", split=split).select(range(n_examples))

    correct = 0
    for example in dataset:
        context = example["ctx"]
        scores = [
            _normalized_loglikelihood(model, tokenizer, context, " " + ending, device)
            for ending in example["endings"]
        ]
        if scores.index(max(scores)) == int(example["label"]):
            correct += 1

    backend.unload()
    return QualityResult(
        model_key=spec.key,
        task="hellaswag",
        accuracy=correct / len(dataset),
        n_examples=len(dataset),
        quantization=quantization,
        note="length-normalized loglikelihood, subset only",
    )
