"""Batched target-coupled tree execution with per-request KV compaction."""
from dataclasses import replace
import hashlib
import math
import random
import time

import torch
from transformers import DynamicCache

from tlar_adaptive_tree import (AdaptiveTree, Candidate, Config, Node, Plan, Verified,
                                kv_commit_indices, tree_layout, union_tree)
from tlar_hf_tree import select_cache
from scripts.compare_retrieval_baselines import stand_tree_prefixes, round_robin_unique_prefixes

PAPER_MODES = ('small_draft', 'stand', 'tlar', 'budgeted_union')


def stream_seed(seed, request_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{request_id}'.encode()).digest()[:8], 'big')


def probabilities(logits, temperature, top_p):
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError('Require finite temperature >= 0 and 0 < top_p <= 1')
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).double()
    p = torch.softmax(logits.double() / temperature, dim=-1)
    if top_p < 1:
        values, order = p.sort(descending=True, stable=True)
        keep = values.cumsum(-1) - values < top_p
        values = values * keep
        p = torch.zeros_like(p).scatter(-1, order, values)
        p /= p.sum(-1, keepdim=True)
    return p


def coupled_verify(plan, logits, uniforms, limit, depth, temperature=.6, top_p=.95, eos_ids=()):
    """Consume one shared uniform per committed token; end at full tree depth."""
    if limit < 1 or len(uniforms) < limit or len(logits) != len(plan.nodes) + 1:
        raise ValueError('Invalid verification inputs')
    children = {(n.parent, n.prefix[-1]): i for i, n in enumerate(plan.nodes)}
    emitted, accepted, parent = [], [], -1
    correction = False
    while len(emitted) < limit:
        u = uniforms[len(emitted)]
        if not 0 <= u < 1:
            raise ValueError('Uniforms must lie in [0, 1)')
        cdf = probabilities(logits[parent + 1], temperature, top_p).cumsum(-1)
        token = min(int(torch.searchsorted(cdf, cdf.new_tensor(u), right=True)), len(cdf)-1)
        emitted.append(token)
        child = children.get((parent, token))
        if child is None:
            correction = True
            break
        accepted.append(child)
        parent = child
        if token in eos_ids or len(plan.nodes[child].prefix) >= depth:
            break
    return Verified(tuple(emitted), tuple(accepted), correction)


PROFILE_RANGES = False


def span(name):
    from contextlib import nullcontext
    return torch.profiler.record_function(name) if PROFILE_RANGES else nullcontext()


@torch.inference_mode()
def batch_forward(backend, tokens, caches, positions=None, allowed=None):
    """Pack ragged caches and queries into one model call with explicit masks."""
    if len(tokens) != len(caches) or any(not row for row in tokens):
        raise ValueError('Each request requires a nonempty query')
    batch = len(tokens)
    lengths = [c.get_seq_length() for c in caches]
    past, query = max(lengths), max(map(len, tokens))
    device, dtype = backend.device, backend.dtype
    with span("tlar/kv_pack"):
        packed = DynamicCache()
        if past:
            exemplar = next(c for c in caches if c.get_seq_length())
            for layer_id, layer in enumerate(exemplar.layers):
                shape = (batch, layer.keys.shape[1], past, layer.keys.shape[3])
                keys = torch.zeros(shape, device=device, dtype=dtype)
                values = torch.zeros_like(keys)
                for i, c in enumerate(caches):
                    if lengths[i]:
                        keys[i, :, past-lengths[i]:] = c.layers[layer_id].keys[0]
                        values[i, :, past-lengths[i]:] = c.layers[layer_id].values[0]
                packed.update(keys, values, layer_id)
    with span("tlar/metadata_transfer"):
        ids = torch.zeros((batch, query), device=device, dtype=torch.long)
        pos = torch.zeros_like(ids)
        mask = torch.full((batch, 1, query, past+query), float('-inf'), device=device, dtype=dtype)
        for i, row in enumerate(tokens):
            n, q = lengths[i], len(row)
            ids[i, :q] = torch.tensor(row, device=device)
            ps = positions[i] if positions is not None else list(range(n, n+q))
            pos[i, :q] = torch.tensor(ps, device=device)
            local = allowed[i] if allowed is not None else [
                [True]*n + [j <= k for j in range(q)] for k in range(q)]
            local = torch.tensor(local, dtype=torch.bool, device=device)
            mask[i, 0, :q, past-n:past] = torch.where(local[:, :n], 0., float('-inf'))
            mask[i, 0, :q, past:past+q] = torch.where(local[:, n:], 0., float('-inf'))
            # Padding query outputs are discarded; give each a finite attention row.
            for k in range(q, query):
                mask[i, 0, k, past+k] = 0
    with span("tlar/model_forward"):
        result = backend.model(input_ids=ids, position_ids=pos, attention_mask=mask,
                               past_key_values=packed, use_cache=True, return_dict=True)
    with span("tlar/kv_unpack"):
        logits, next_caches = [], []
        for i, row in enumerate(tokens):
            n, q = lengths[i], len(row)
            indices = list(range(past-n, past)) + list(range(past, past+q))
            cache = DynamicCache()
            for j, layer in enumerate(result.past_key_values.layers):
                ix = torch.tensor(indices, device=layer.keys.device)
                cache.update(layer.keys[i:i+1].index_select(-2, ix),
                             layer.values[i:i+1].index_select(-2, ix), j)
            next_caches.append(cache)
            logits.append(result.logits[i, :q])
    return logits, next_caches


def compose_plan(controller, request_id, history, small, mode, node_budget, seed):
    cfg = controller.config
    use_tlar = mode in ('tlar', 'budgeted_union')
    if use_tlar:
        plan = controller.propose(request_id, history, small)
    else:
        plan = Plan(request_id, len(history), 0., False, False, 0,
                    tuple(small), (), union_tree(small, ()))
    candidates = plan.candidates if use_tlar else ()
    retrieval = []
    for c in candidates:
        for n in range(1, len(c.tokens)+1):
            if c.tokens[:n] not in retrieval:
                retrieval.append(c.tokens[:n])
    stand = stand_tree_prefixes(history, len(history), cfg.k_max, cfg.depth,
              2, 8, 1., stream_seed(seed, f'{request_id}:{len(history)}')) if (
              mode in ('stand', 'budgeted_union') and len(history) > cfg.history_gate) else []
    if not hasattr(controller, 'source_scores'):controller.source_scores={}
    if not hasattr(controller, 'source_paths'):controller.source_paths={}
    scores=controller.source_scores.get(request_id, (0.,0.))
    first,second=(stand,retrieval) if scores[0]>=scores[1] else (retrieval,stand)
    prefixes = round_robin_unique_prefixes(first, second, node_budget) if mode == 'budgeted_union' else stand + retrieval
    selected=set(prefixes)
    controller.source_paths[request_id]=(set(stand)&selected,set(retrieval)&selected)
    base = union_tree(small, ())
    all_paths = sorted({n.prefix for n in base} | set(prefixes), key=lambda p:(len(p),p))
    indices = {p:i for i,p in enumerate(all_paths)}
    small_paths = {n.prefix for n in base}
    retrieval_paths = set(retrieval)
    nodes = tuple(Node(p, indices.get(p[:-1],-1), p in small_paths,
                       p in retrieval_paths) for p in all_paths)
    # All paths are prefix closed; SmallDraft paths remain reserved separately.
    plan = replace(plan, active=plan.active and use_tlar, candidates=candidates,
                   width=plan.width if use_tlar else 0, nodes=nodes)
    controller.pending[request_id] = plan
    return plan


def observe_sources(controller, request_id, emitted):
    if not hasattr(controller,'source_paths') or request_id not in controller.source_paths:return
    scores=controller.source_scores.get(request_id,(0.,0.))
    decay=2**(-1/controller.config.half_life)
    groups=controller.source_paths[request_id]
    controller.source_scores[request_id]=tuple(
        (decay*score+(1-decay)*int((emitted[0],) in paths))*decay**(len(emitted)-1)
        for score,paths in zip(scores,groups))


@torch.inference_mode()
def decode_batch(target, draft, prompts, budget, mode, config=Config(), *,
                 seed=0, request_ids=None, temperature=.6, top_p=.95,
                 node_budget=8, eos_ids=(), uniforms=None):
    if mode not in (*PAPER_MODES, 'vanilla') or not prompts or any(not p for p in prompts):
        raise ValueError('Invalid mode or empty prompt')
    if budget < 1 or node_budget < 1:
        raise ValueError('Positive output and node budgets required')
    ids = list(request_ids) if request_ids is not None else [str(i) for i in range(len(prompts))]
    if len(ids) != len(prompts) or len(set(ids)) != len(ids):
        raise ValueError('Request IDs must be unique and match batch size')
    if uniforms is None:
        uniforms = []
        for key in ids:
            rng = random.Random(stream_seed(seed,key))
            uniforms.append([rng.random() for _ in range(budget)])
    if len(uniforms) != len(prompts) or any(len(u)<budget for u in uniforms):
        raise ValueError('Insufficient shared uniforms')
    def sync():
        if target.device.type == 'cuda': torch.cuda.synchronize(target.device)
    sync(); start = time.perf_counter()
    controller = AdaptiveTree(config)
    generated = [[] for _ in prompts]
    records = [[] for _ in prompts]
    caches = [DynamicCache() for _ in prompts]
    draft_caches = [DynamicCache() for _ in prompts]
    draft_tokens = [[] for _ in prompts]
    prefill_ids = [i for i,p in enumerate(prompts) if len(p)>1]
    if prefill_ids:
        _, cs = batch_forward(target,[prompts[i][:-1] for i in prefill_ids], [caches[i] for i in prefill_ids])
        for i,c in zip(prefill_ids,cs): caches[i]=c
    active = list(range(len(prompts)))
    while active:
        prefixes = [list(prompts[i])+generated[i] for i in active]
        small = [[] for _ in active]
        with span("tlar/draft"):
            if mode != 'vanilla':
                pending=[]
                for i,p in zip(active,prefixes):
                    common=0
                    for a,b in zip(draft_tokens[i],p[:-1]):
                        if a!=b:break
                        common+=1
                    draft_caches[i].crop(common)
                    draft_tokens[i]=draft_tokens[i][:common]
                    pending.append(p[common:])
                for _ in range(min(config.depth,max(budget-len(generated[i]) for i in active))):
                    ls,cs=batch_forward(draft,pending,[draft_caches[i] for i in active])
                    for k,(i,c,l) in enumerate(zip(active,cs,ls)):
                        draft_caches[i]=c;draft_tokens[i].extend(pending[k])
                        tok=int(l[-1].argmax());small[k].append(tok);pending[k]=[tok]
        with span("tlar/retrieval_tree"):
            plans=[compose_plan(controller,ids[i],generated[i],s,mode,node_budget,seed)
                   for i,s in zip(active,small)]
        with span("tlar/target_verification"):
            layouts=[tree_layout(p,len(h)) for p,h in zip(plans,prefixes)]
            queries=[[h[-1]]+[n.prefix[-1] for n in p.nodes] for p,h in zip(plans,prefixes)]
            ls,cs=batch_forward(target,queries,[caches[i] for i in active],
                                [x[0] for x in layouts],[x[1] for x in layouts])
        next_active=[]
        for i,p,h,l,c in zip(active,plans,prefixes,ls,cs):
            n=len(generated[i]);limit=budget-n
            if mode in ('tlar','budgeted_union'): limit=controller.commit_limit(p,limit)
            v=coupled_verify(p,l,uniforms[i][n:n+limit],limit,config.depth,temperature,top_p,eos_ids)
            with span("tlar/accepted_path_cache"):
                keep=kv_commit_indices(len(h),v)[:len(h)+len(v.tokens)-1]
                select_cache(c,keep);caches[i]=c
            if mode in ('tlar','budgeted_union'):
                # Credit TLAR only when a selected TLAR path was actually verified.
                selected={node.prefix for node in p.nodes if node.retrieval}
                observed=replace(p,candidates=tuple(cand for cand in p.candidates if (cand.tokens[0],) in selected))
                controller.pending[ids[i]]=observed
                event=controller.observe(observed,v.tokens)
            else:
                event={'history_length':n,'active':p.active,'width':p.width}
                controller.finish(ids[i])
            event.update(emitted=list(v.tokens),small=list(p.small),tree_nodes=len(p.nodes),
                         accepted_nodes=list(v.accepted_nodes),correction=v.bonus,
                         cache_tokens=c.get_seq_length(),candidate_starts=[x.start for x in p.candidates])
            observe_sources(controller,ids[i],v.tokens)
            records[i].append(event);generated[i].extend(v.tokens)
            if len(generated[i])<budget and v.tokens[-1] not in eos_ids:next_active.append(i)
        active=next_active
    sync();elapsed=time.perf_counter()-start
    return generated,records,elapsed
