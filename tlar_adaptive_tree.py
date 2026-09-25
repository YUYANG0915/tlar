"""Causal retrieval, per-request control, union trees and target-path verification."""

from __future__ import annotations

from dataclasses import dataclass
import math
import itertools
from collections import defaultdict
from typing import Sequence


@dataclass(frozen=True)
class Config:
    context: int = 4
    edits: int = 1
    depth: int = 4
    history_gate: int = 512
    k_min: int = 1
    k_max: int = 4
    half_life: float = 32
    rho_min: float = 0.05
    probe_interval: int = 32

    def __post_init__(self):
        if not (self.context >= 1 and 0 <= self.edits <= self.context):
            raise ValueError("Invalid context or Hamming tolerance")
        if self.depth < 1 or self.history_gate < 0:
            raise ValueError("Invalid depth or history gate")
        if not 1 <= self.k_min <= self.k_max:
            raise ValueError("Require 1 <= k_min <= k_max")
        if not math.isfinite(self.half_life) or self.half_life <= 0:
            raise ValueError("half_life must be finite and positive")
        if not 0 <= self.rho_min <= 1 or self.probe_interval < 1:
            raise ValueError("Invalid threshold or probe interval")


@dataclass(frozen=True)
class Candidate:
    start: int
    tokens: tuple[int, ...]


def retrieve(history: Sequence[int], config: Config, width: int) -> tuple[Candidate, ...]:
    """Zero-based j: c <= j <= len(history)-d, exactly Eq. (1)."""
    if width < 0 or width > config.k_max:
        raise ValueError("Invalid candidate width")
    if not width or len(history) < config.context + config.depth:
        return ()
    n, c, d = len(history), config.context, config.depth
    context = tuple(history[n - c:])
    found = []
    for j in range(n - d, c - 1, -1):
        distance = sum(a != b for a, b in zip(history[j - c:j], context))
        if distance <= config.edits:
            found.append(Candidate(j, tuple(history[j:j + d])))
            if len(found) == width:
                break
    return tuple(found)


class HistoryIndex:
    """Insert complete committed continuations and rank matches by recency."""
    def __init__(self, config):
        self.config = config
        self.positions = defaultdict(list)
        self.inserted = config.context - 1
        self.history = ()

    def keys(self, context):
        for mask in itertools.combinations(range(len(context)), self.config.edits):
            key = list(context)
            for i in mask:
                key[i] = None
            yield tuple(key)

    def query(self, history, width):
        cfg = self.config
        if tuple(history[:len(self.history)]) != self.history:
            raise ValueError("Committed history changed")
        self.history = tuple(history)
        for j in range(self.inserted + 1, len(history) - cfg.depth + 1):
            for key in self.keys(history[j-cfg.context:j]):
                self.positions[key].append(j)
            self.inserted = j
        if width == 0 or len(history) < cfg.context:
            return ()
        found = set()
        for key in self.keys(history[-cfg.context:]):
            found.update(self.positions.get(key, ()))
        return tuple(Candidate(j, tuple(history[j:j+cfg.depth]))
                     for j in sorted(found, reverse=True)[:width])


@dataclass(frozen=True)
class Node:
    prefix: tuple[int, ...]
    parent: int  # -1 denotes the already-generated root.
    small: bool
    retrieval: bool


def union_tree(small: Sequence[int], candidates: Sequence[Candidate]) -> tuple[Node, ...]:
    origins: dict[tuple[int, ...], list[bool]] = {}
    for origin, tokens in [(0, tuple(small))] + [(1, c.tokens) for c in candidates]:
        for depth in range(1, len(tokens) + 1):
            origins.setdefault(tokens[:depth], [False, False])[origin] = True
    prefixes = sorted(origins, key=lambda p: (len(p), p))
    indices = {p: i for i, p in enumerate(prefixes)}
    return tuple(Node(p, indices.get(p[:-1], -1), *origins[p]) for p in prefixes)


@dataclass(frozen=True)
class Plan:
    request_id: str
    history_length: int
    rho_before: float
    active: bool
    probe: bool
    width: int
    small: tuple[int, ...]
    candidates: tuple[Candidate, ...]
    nodes: tuple[Node, ...]


@dataclass
class State:
    rho: float = 0.0
    last_attempt_success: bool = False
    next_history_length: int = 0


class AdaptiveTree:
    """Per-request state. Feedback must arrive before that request proposes again.

    The clock is generated-token position, not batch row or verification call.
    The first eligible position (history=tau+1) is a probe, then every P tokens.
    A backend must cap each commit using commit_limit() so probes are not skipped.
    Skipped, non-probe positions decay the score with m=0.
    """

    def __init__(self, config: Config = Config()):
        self.config = config
        self.states: dict[str, State] = {}
        self.pending: dict[str, Plan] = {}
        self.indices: dict[str, HistoryIndex] = {}

    def propose(self, request_id: str, history: Sequence[int], small: Sequence[int]) -> Plan:
        if request_id in self.pending:
            raise RuntimeError("Missing verified feedback for this request")
        cfg = self.config
        n = len(history)
        state = self.states.setdefault(request_id, State())
        if n < state.next_history_length:
            raise ValueError("History moved backwards; reset state for a new request")
        if state.next_history_length and n != state.next_history_length:
            raise ValueError("History advance does not match the verified commit")
        rho = state.rho * 2.0 ** (-(n - state.next_history_length) / cfg.half_life)
        eligible = n > cfg.history_gate
        probe = eligible and (n - cfg.history_gate - 1) % cfg.probe_interval == 0
        active = eligible and (rho >= cfg.rho_min or state.last_attempt_success or probe)
        width = 0
        if active:
            width = cfg.k_min if probe else min(cfg.k_max, max(cfg.k_min, math.ceil(cfg.k_max * rho)))
        index = self.indices.setdefault(request_id, HistoryIndex(cfg))
        candidates = index.query(history, width)
        plan = Plan(request_id, n, rho, active, probe, width, tuple(small),
                    candidates, union_tree(small, candidates))
        self.pending[request_id] = plan
        return plan

    def commit_limit(self, plan: Plan, remaining: int) -> int:
        if remaining < 1:
            raise ValueError("No remaining output budget")
        cfg = self.config
        first = cfg.history_gate + 1
        n = plan.history_length
        next_probe = first if n < first else first + ((n - first) // cfg.probe_interval + 1) * cfg.probe_interval
        return min(remaining, next_probe - n)

    def observe(self, plan: Plan, verified_tokens: Sequence[int]) -> dict:
        """Call only with committed target-verified tokens, never draft guesses."""
        if self.pending.get(plan.request_id) is not plan:
            raise RuntimeError("Stale, foreign, or duplicate feedback")
        emitted = tuple(verified_tokens)
        if not emitted or len(emitted) > self.commit_limit(plan, len(emitted)):
            raise ValueError("Empty commit or skipped probe position")
        success = bool(plan.candidates and any(c.tokens[0] == emitted[0] for c in plan.candidates))
        decay = 2.0 ** (-1.0 / self.config.half_life)
        rho = decay * plan.rho_before + (1.0 - decay) * int(plan.active and success)
        # No retrieval was attempted at positions skipped by the verified chunk.
        rho *= decay ** (len(emitted) - 1)
        state = self.states[plan.request_id]
        state.rho = rho
        if plan.active:
            state.last_attempt_success = success
        state.next_history_length = plan.history_length + len(emitted)
        del self.pending[plan.request_id]
        return {"request_id": plan.request_id, "history_length": plan.history_length,
                "active": plan.active, "probe": plan.probe, "width": plan.width,
                "rho_before": plan.rho_before, "rho_after": rho,
                "retrieval_hit": success, "committed_tokens": len(emitted),
                "small_nodes": sum(n.small for n in plan.nodes),
                "retrieval_nodes": sum(n.retrieval for n in plan.nodes),
                "union_nodes": len(plan.nodes),
                "added_nodes": sum(not n.small for n in plan.nodes)}

    def finish(self, request_id: str) -> None:
        self.pending.pop(request_id, None)
        self.states.pop(request_id, None)
        self.indices.pop(request_id, None)


def tree_layout(plan: Plan, prefix_length: int) -> tuple[list[int], list[list[bool]]]:
    """Root + node positions and allowed-attention mask for a KV backend.

    Cache contains prefix_length-1 tokens. Root is the final committed token.
    Columns are cached prefix, root, and flattened tree nodes. Siblings and
    descendants must never be visible to a node, even if earlier in the buffer.
    """
    if prefix_length < 1:
        raise ValueError("Tree verification requires a nonempty committed prefix")
    paths = [()] + [n.prefix for n in plan.nodes]
    positions = [prefix_length - 1 + len(p) for p in paths]
    mask = []
    for path in paths:
        ancestors = [path[:len(p)] == p for p in paths]
        mask.append([True] * (prefix_length - 1) + ancestors)
    return positions, mask


@dataclass(frozen=True)
class Verified:
    tokens: tuple[int, ...]
    accepted_nodes: tuple[int, ...]
    bonus: bool


def verify_greedy(plan: Plan, next_tokens: Sequence[int], limit: int,
                  eos_token_id: int | None = None) -> Verified:
    """next_tokens[0] is root argmax; [i+1] is argmax after node i.

    Predictions must come from one properly masked target tree forward pass.
    No candidate gets priority over the target's greedy next token.
    """
    if len(next_tokens) != len(plan.nodes) + 1 or limit < 1:
        raise ValueError("Missing node predictions or invalid commit limit")
    children = {(n.parent, n.prefix[-1]): i for i, n in enumerate(plan.nodes)}
    parent = -1
    emitted, accepted = [], []
    bonus = False
    while len(emitted) < limit:
        token = int(next_tokens[parent + 1])
        emitted.append(token)
        child = children.get((parent, token))
        if child is None:
            bonus = True
            break
        accepted.append(child)
        parent = child
        if len(plan.nodes[child].prefix) >= max((len(n.prefix) for n in plan.nodes), default=0):
            break
        if token == eos_token_id:
            break
    return Verified(tuple(emitted), tuple(accepted), bonus)


def kv_commit_indices(prefix_length: int, verified: Verified) -> tuple[int, ...]:
    """Keep committed prefix/root and the accepted path; bonus is not cached yet."""
    if prefix_length < 1:
        raise ValueError("Invalid prefix length")
    return tuple(range(prefix_length)) + tuple(prefix_length + i for i in verified.accepted_nodes)
