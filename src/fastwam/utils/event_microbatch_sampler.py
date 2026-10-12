"""Order-only task interleaving, preserving the exact selected sample multiset."""
from collections import Counter

import torch


def shuffle_event_microbatches(indices, *, segment_groups, batch_size, num_processes, generator):
    """Emit rank-sized batches, mixing tasks/episodes across each global step.

    Full segment chunks retain one shared memory. Segment remainders are packed
    within their task (the model already handles multiple groups per rank).
    No real sample is added, removed, or replaced. Global padding remains at
    the very end and belongs to the caller, not to individual segments.
    """
    if batch_size <= 0 or num_processes <= 0:
        raise ValueError("Batch size and world size must be positive.")
    if any(index < 0 for index in indices):
        raise ValueError("Pass real samples only; append global padding afterward.")
    lookup = {
        int(anchor): (task, group)
        for task, groups in segment_groups.items()
        for group, anchors in groups.items()
        for anchor in anchors
    }
    selected = {}
    for index in indices:
        task, group = lookup[index]
        selected.setdefault(task, {}).setdefault(group, []).append(index)

    pools, final_remainders = {}, []
    for task in sorted(selected):
        pool, tails = [], []
        for group, values in selected[task].items():
            stop = len(values) // batch_size * batch_size
            for start in range(0, stop, batch_size):
                pool.append((values[start:start + batch_size], frozenset({group})))
            tails.extend(values[stop:])
        stop = len(tails) // batch_size * batch_size
        for start in range(0, stop, batch_size):
            chunk = tails[start:start + batch_size]
            pool.append((chunk, frozenset(lookup[index][1] for index in chunk)))
        final_remainders.extend(tails[stop:])
        permutation = torch.randperm(len(pool), generator=generator).tolist()
        pools[task] = [pool[i] for i in permutation]

    # Weighted deficit scheduling: equal task quotas give 4+4 ranks at world=8;
    # a mid-epoch transition distributes its remaining quotas proportionally.
    totals = {task: len(pool) for task, pool in pools.items()}
    total = sum(totals.values())
    used = Counter()
    output = []
    for start in range(0, total, num_processes):
        rank_batches, seen_groups = [], set()
        for slot in range(start, min(start + num_processes, total)):
            active = [task for task in sorted(pools) if pools[task]]
            task = max(active, key=lambda t: (slot + 1) * totals[t] - used[t] * total)
            pool = pools[task]
            position = len(pool) - 1
            # Prefer different episodes/segments across ranks in one update.
            for candidate in range(len(pool) - 1, -1, -1):
                if seen_groups.isdisjoint(pool[candidate][1]):
                    position = candidate
                    break
            pool[position], pool[-1] = pool[-1], pool[position]
            chunk, groups = pool.pop()
            seen_groups.update(groups)
            rank_batches.append(chunk)
            used[task] += 1
        # Do not pin either task to a particular GPU.
        rank_order = torch.randperm(len(rank_batches), generator=generator).tolist()
        for rank in rank_order:
            output.extend(rank_batches[rank])
    output.extend(final_remainders)
    if Counter(output) != Counter(indices):
        raise AssertionError("Microbatch shuffle altered the selected sample multiset.")
    return output
