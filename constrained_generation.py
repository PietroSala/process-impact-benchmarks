"""Weighted generation conditioned on reachable XOR + loop complexity targets.

Unlike whole-process rejection, this sampler excludes expansions that exceed
the targets or make the requested independent count unreachable. It keeps the
requested relative operator weights among the remaining choices. The resulting
sampling distribution differs from the original rejection sampler.
"""

from dataclasses import dataclass, field
import random

from random_diagram_generation import (
    DEFAULT_PROBABILITIES, REPLACEMENTS, _validate_probabilities,
    replace_underscores, weighted_choice,
)


@dataclass(eq=False)
class _Region:
    depth: int
    loop_allowed: bool
    capacity: int
    parent: object = field(default=None, repr=False)
    kind: str = "task"
    children: list = field(default_factory=list)
    nested: int = 0
    independent: int = 0

    @property
    def metrics(self):
        return self.nested, self.independent, self.capacity


def _combine(kind, children, limit):
    nested = max(m[0] for m in children)
    if kind in {"xor", "loop"}:
        return nested + 1, max(1, *(m[1] for m in children)), max(1, *(m[2] for m in children))
    return nested, sum(m[1] for m in children), min(limit, sum(m[2] for m in children))


def _root_metrics(region, replacement_metrics, limit):
    """Evaluate a proposed change without modifying the partial process."""
    while region.parent is not None:
        parent = region.parent
        replacement_metrics = _combine(
            parent.kind,
            [replacement_metrics if child is region else child.metrics for child in parent.children],
            limit,
        )
        region = parent
    return replacement_metrics


def _expression(region):
    if region.kind == "task":
        return "_"
    if region.kind == "loop":
        return f"(! {_expression(region.children[0])})"
    operator = {"xor": "^", "parallel": "||", "sequential": ","}[region.kind]
    return f"({_expression(region.children[0])} {operator} {_expression(region.children[1])})"


def generate_constrained_process(probabilities, target_max_nested_xor,
                                 target_max_independent_xor, number_of_replacements,
                                 forbidden_processes):
    """Generate a process with at least one loop, or None within the budget.

    Leaves at the nesting limit, or with no room for another decision, are
    excluded from expansion. Each candidate must retain enough independent
    capacity to reach the target. XOR always unlocks nested loops; a parallel
    or sequence preserves the current loop restriction.
    """
    weights = DEFAULT_PROBABILITIES if probabilities is None else tuple(probabilities)
    if len(weights) == 3:
        weights += (0.0,)
    _validate_probabilities(REPLACEMENTS, weights)
    nested_target = target_max_nested_xor
    independent_target = target_max_independent_xor
    if nested_target < 1 or independent_target < 1:
        raise ValueError("Complexity targets must be positive")
    if weights[3] == 0:
        return None
    # The benchmark requires positive XOR and connector weights so that the
    # independent capacity of an unsaturated leaf is exactly the target cap.
    if weights[0] == 0:
        raise ValueError("Constrained generation requires positive XOR probability")
    connectors = weights[1] + weights[2] > 0
    if independent_target > 1 and not connectors:
        raise ValueError("Independent regions require parallel or sequence probability")

    def capacity(depth):
        return (independent_target if connectors else 1) if depth < nested_target else 0

    root = _Region(0, True, capacity(0))
    leaves = [root]
    loop_count = 0
    for _ in range(number_of_replacements):
        candidates = []
        for leaf in leaves:
            if leaf.depth >= nested_target:
                continue
            decision_metrics = (1, 1, max(1, capacity(leaf.depth + 1)))
            proposed = _root_metrics(leaf, decision_metrics, independent_target)
            if proposed[0] > nested_target or proposed[1] > independent_target:
                continue
            # Connectors preserve current metrics and capacity. They may be
            # essential before a choice (e.g. nesting=1, independent=10).
            allowed = [i for i in (1, 2) if weights[i] > 0]
            if proposed[2] >= independent_target:
                allowed.append(0)
                if leaf.loop_allowed:
                    allowed.append(3)
            if allowed:
                candidates.append((leaf, sorted(allowed)))
        if not candidates:
            return None

        leaf, allowed = random.choice(candidates)
        total = sum(weights[i] for i in allowed)
        selected = weighted_choice(allowed, [weights[i] / total for i in allowed])
        leaf.kind = ("xor", "parallel", "sequential", "loop")[selected]
        decision = selected in (0, 3)
        child_depth = leaf.depth + int(decision)
        child_loop_allowed = selected == 0 if decision else leaf.loop_allowed
        leaf.children = [
            _Region(child_depth, child_loop_allowed, capacity(child_depth), parent=leaf)
            for _ in range(1 if selected == 3 else 2)
        ]
        leaves.remove(leaf)
        leaves.extend(leaf.children)
        loop_count += selected == 3
        region = leaf
        while region is not None:
            region.nested, region.independent, region.capacity = _combine(
                region.kind, [child.metrics for child in region.children], independent_target,
            )
            region = region.parent

        if (root.nested, root.independent) == (nested_target, independent_target):
            if not loop_count:
                return None
            expression = replace_underscores(_expression(root))
            return expression if expression not in forbidden_processes else None
    return None
