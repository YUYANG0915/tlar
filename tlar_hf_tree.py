"""Single-request greedy HF execution and KV-cache helpers."""

from dataclasses import replace
import time

import torch
from transformers import DynamicCache

from tlar_adaptive_tree import (
    AdaptiveTree, Candidate, Config, Plan, retrieve, tree_layout, union_tree,
    verify_greedy, kv_commit_indices,
)


def select_cache(cache, indices):
    """Discard siblings and the last committed token (next round's root)."""
    for layer in cache.layers:
        if type(layer).__name__ != "DynamicLayer":
            raise RuntimeError("Reference backend requires full DynamicLayer caches")
        index = torch.tensor(indices, dtype=torch.long, device=layer.keys.device)
        layer.keys = layer.keys.index_select(-2, index)
        layer.values = layer.values.index_select(-2, index)
    if cache.get_seq_length() != len(indices):
        raise RuntimeError("Cache selection did not preserve sequence length")


class Backend:
    def __init__(self, model):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    @torch.inference_mode()
    def forward(self, tokens, cache, positions=None, allowed=None):
        kwargs = dict(
            input_ids=torch.tensor([tokens], dtype=torch.long, device=self.device),
            past_key_values=cache, use_cache=True, return_dict=True,
        )
        if positions is not None:
            kwargs["position_ids"] = torch.tensor([positions], device=self.device)
        if allowed is not None:
            mask = torch.tensor(allowed, dtype=torch.bool, device=self.device)[None, None]
            kwargs["attention_mask"] = torch.zeros(mask.shape, dtype=self.dtype,
                                                   device=self.device).masked_fill(~mask, float("-inf"))
        result = self.model(**kwargs)
        if result.past_key_values is not cache:
            raise RuntimeError("Backend unexpectedly replaced the cache")
        return result.logits[0]

    def prefill(self, tokens):
        cache = DynamicCache()
        if tokens:
            self.forward(tokens, cache)
        return cache

    def tree(self, prefix, cache, plan):
        if cache.get_seq_length() != len(prefix) - 1:
            raise RuntimeError("Target cache/root boundary mismatch")
        positions, allowed = tree_layout(plan, len(prefix))
        tokens = [prefix[-1]] + [node.prefix[-1] for node in plan.nodes]
        return self.forward(tokens, cache, positions, allowed)


class Draft:
    def __init__(self, backend):
        self.backend = backend
        self.cache = DynamicCache()
        self.cached_tokens = []

    def propose(self, prefix, depth):
        common = 0
        for a, b in zip(self.cached_tokens, prefix[:-1]):
            if a != b:
                break
            common += 1
        self.cache.crop(common)
        self.cached_tokens = self.cached_tokens[:common]
        pending = list(prefix[common:])
        draft = []
        for _ in range(depth):
            logits = self.backend.forward(pending, self.cache)
            self.cached_tokens.extend(pending)
            token = int(logits[-1].argmax())
            draft.append(token)
            pending = [token]
        return draft


MODES = ("vanilla", "small_draft", "fixed_union", "adaptive_union", "exact_union")


def make_plan(controller, request_id, history, small, mode):
    plan = controller.propose(request_id, history, small)
    if mode == "adaptive_union":
        return plan
    cfg = controller.config
    width = cfg.k_max if mode in ("fixed_union", "exact_union") and len(history) > cfg.history_gate else 0
    candidates = retrieve(history, replace(cfg, edits=0) if mode == "exact_union" else cfg, width)
    plan = replace(plan, active=bool(width), probe=False, width=width,
                   candidates=candidates, nodes=union_tree(small, candidates))
    controller.pending[request_id] = plan
    return plan


@torch.inference_mode()
def decode(target, draft_backend, prompt, budget, mode, config=Config(), eos_ids=()):
    if mode not in MODES or not prompt or budget < 1:
        raise ValueError("Invalid decoding request")
    controller = AdaptiveTree(config)
    draft = Draft(draft_backend) if mode != "vanilla" else None
    generated, rounds = [], []
    if target.device.type == "cuda":
        torch.cuda.synchronize(target.device)
    start = time.perf_counter()
    cache = target.prefill(prompt[:-1])
    while len(generated) < budget:
        prefix = list(prompt) + generated
        small = draft.propose(prefix, min(config.depth, budget - len(generated))) if draft else []
        plan = make_plan(controller, "request", generated, small, mode)
        limit = budget - len(generated)
        if mode == "adaptive_union":
            limit = controller.commit_limit(plan, limit)
        predictions = target.tree(prefix, cache, plan).argmax(-1).tolist()
        verified = verify_greedy(plan, predictions, limit)
        emitted = list(verified.tokens)
        for i, token in enumerate(emitted):
            if token in eos_ids:
                emitted = emitted[:i + 1]
                break
        # The last emitted token is intentionally recomputed as the next root.
        keep = kv_commit_indices(len(prefix), verified)[:len(prefix) + len(emitted) - 1]
        select_cache(cache, keep)
        if mode == "adaptive_union":
            record = controller.observe(plan, emitted)
        else:
            controller.finish("request")
            record = dict(history_length=len(generated), active=plan.active,
                          probe=False, width=plan.width, committed_tokens=len(emitted))
        record.update(
            candidate_starts=[c.start for c in plan.candidates],
            candidate_tokens=[list(c.tokens) for c in plan.candidates],
            small=list(plan.small), tree_nodes=len(plan.nodes),
            accepted_nodes=list(verified.accepted_nodes[:len(emitted)]),
            emitted=emitted, cache_tokens=cache.get_seq_length(),
        )
        rounds.append(record)
        generated.extend(emitted)
        if emitted[-1] in eos_ids:
            break
    if target.device.type == "cuda":
        torch.cuda.synchronize(target.device)
    elapsed = time.perf_counter() - start
    return generated, rounds, elapsed


@torch.inference_mode()
def check_tree_logits(backend, prompt, atol=0.05, rtol=0.01):
    """Check siblings, full prefixes, path compaction, and the next root forward."""
    cache = backend.prefill(prompt[:-1])
    root_next = int(backend.forward([prompt[-1]], cache)[-1].argmax())
    cache.crop(len(prompt) - 1)
    vocab = backend.model.get_output_embeddings().weight.shape[0]
    other = (root_next + 1) % vocab
    small = (root_next, other)
    candidates = (Candidate(0, (other, root_next)),)
    plan = Plan("check", 0, 0.0, True, False, 1, small, candidates,
                union_tree(small, candidates))
    logits = backend.tree(prompt, cache, plan)
    max_error = 0.0
    for row, path in enumerate([()] + [node.prefix for node in plan.nodes]):
        reference = backend.forward(list(prompt) + list(path), DynamicCache())[-1]
        delta = float((logits[row].float() - reference.float()).abs().max())
        max_error = max(max_error, delta)
        torch.testing.assert_close(logits[row], reference, atol=atol, rtol=rtol)
        if int(logits[row].argmax()) != int(reference.argmax()):
            raise AssertionError("Tree/sequential greedy prediction mismatch")
    verified = verify_greedy(plan, logits.argmax(-1).tolist(), 3)
    prefix = list(prompt) + list(verified.tokens)
    select_cache(cache, kv_commit_indices(len(prompt), verified)[:len(prefix) - 1])
    after = backend.forward([prefix[-1]], cache)[-1]
    reference = backend.forward(prefix, DynamicCache())[-1]
    torch.testing.assert_close(after, reference, atol=atol, rtol=rtol)
    if int(after.argmax()) != int(reference.argmax()):
        raise AssertionError("Committed-cache greedy prediction mismatch")
    return dict(max_logit_error=max_error, passed=True)
