import random

import pytest
from lark import Tree

from constrained_generation import generate_constrained_process
from sese_diagram import PARSER, get_tasks
from stats import max_independent_xor, max_nested_xor


WEIGHTS = (0.3, 0.2, 0.2, 0.3)


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = random.getstate()
    yield
    random.setstate(state)


def assert_valid_process(expression, nested, independent):
    assert "_" not in expression
    assert max_nested_xor(expression) == nested
    assert max_independent_xor(expression) == independent
    tree = PARSER.parse(expression)
    tasks = get_tasks(tree)
    assert tasks == {f"T{index}" for index in range(1, len(tasks) + 1)}
    pending = [(tree, ())]
    loop_count = 0
    while pending:
        node, ancestors = pending.pop()
        if not isinstance(node, Tree):
            continue
        if node.data == "loop":
            loop_count += 1
            for kind in reversed(ancestors):
                if kind in {"xor", "xor_probability"}:
                    break
                assert kind != "loop", expression
        pending.extend((child, (*ancestors, node.data)) for child in node.children)
    assert loop_count > 0


def next_variant(nested, independent, forbidden):
    original_forbidden = forbidden.copy()
    for _ in range(100):
        result = generate_constrained_process(
            WEIGHTS, nested, independent, 500, forbidden,
        )
        assert forbidden == original_forbidden
        if result is not None:
            assert result not in forbidden
            assert_valid_process(result, nested, independent)
            return result
    pytest.fail(f"No new variant for ({nested}, {independent}) in 100 trials")


@pytest.mark.parametrize("nested,independent", [
    (1, 1), (1, 9), (1, 10), (10, 1), (10, 10),
])
def test_ten_distinct_variants_for_challenging_targets(nested, independent):
    random.seed(f"constrained:{nested}:{independent}:2026")
    processes = set()
    for _ in range(10):
        processes.add(next_variant(nested, independent, processes))
    assert len(processes) == 10


def test_repeated_seed_reproduces_variants():
    def generate_sample():
        random.seed(20261008)
        forbidden = set()
        sample = []
        for _ in range(3):
            process = next_variant(3, 4, forbidden)
            forbidden.add(process)
            sample.append(process)
        return sample

    assert generate_sample() == generate_sample()


def test_zero_replacement_budget_returns_none():
    assert generate_constrained_process(WEIGHTS, 1, 1, 0, set()) is None


@pytest.mark.parametrize("weights", [(0.3, 0.3, 0.4), (0.3, 0.3, 0.4, 0)])
def test_no_loop_probability_returns_none(weights):
    assert generate_constrained_process(weights, 1, 1, 500, set()) is None
