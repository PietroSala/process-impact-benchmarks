import random

import pytest
from lark import Tree

from random_batch_generation import generate_multiple_processes
from random_diagram_generation import (
    replace_random_underscore,
    replace_underscores,
    weighted_choice,
)
from sese_diagram import PARSER, get_tasks
from stats import max_independent_xor, max_nested_xor, max_theoretical_pareto_length


LOOP_ONLY = (0, 0, 0, 1)


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = random.getstate()
    yield
    random.setstate(state)


def assert_valid_loop_ancestry(expression):
    """Check each loop against its nearest loop ancestor using parsed paths."""
    tree = PARSER.parse(expression)
    pending = [(tree, ())]
    loop_count = 0
    while pending:
        node, ancestors = pending.pop()
        if not isinstance(node, Tree):
            continue
        if node.data == "loop":
            loop_count += 1
            loop_positions = [
                index for index, ancestor in enumerate(ancestors)
                if ancestor == "loop"
            ]
            if loop_positions:
                between = ancestors[loop_positions[-1] + 1:]
                assert any(kind in {"xor", "xor_probability"} for kind in between), expression
        pending.extend((child, (*ancestors, node.data)) for child in node.children)
    return loop_count


def test_loop_syntax_and_task_names():
    expression = replace_random_underscore("_", LOOP_ONLY)
    assert expression == "(! _)"
    expression = replace_random_underscore(expression, (1, 0, 0, 0))
    named_expression = replace_underscores(expression)
    assert named_expression == "(! (T1 ^ T2))"
    tree = PARSER.parse(named_expression)
    assert tree.data == "loop"
    assert get_tasks(tree) == {"T1", "T2"}


def test_default_probabilities_include_all_four_operators():
    random.seed(31)
    outcomes = {replace_random_underscore("_") for _ in range(200)}
    assert outcomes == {"(_ ^ _)", "(_ || _)", "(_ , _)", "(! _)"}


@pytest.mark.parametrize("probabilities", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.4, 0.3, 0.3)])
def test_legacy_three_probabilities_do_not_generate_loops(probabilities):
    random.seed(42)
    expression = "_"
    for _ in range(30):
        expression = replace_random_underscore(expression, probabilities)
    assert "!" not in expression
    assert len(get_tasks(PARSER.parse(replace_underscores(expression)))) == 31


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ((1, 0, 0, 0), "(_ ^ _)"),
        ((0, 1, 0, 0), "(_ || _)"),
        ((0, 0, 1, 0), "(_ , _)"),
        (LOOP_ONLY, "(! _)"),
    ],
)
def test_four_probabilities_select_each_operator(probabilities, expected):
    assert replace_random_underscore("_", probabilities) == expected


@pytest.mark.parametrize(
    "probabilities",
    [
        (),
        (0.5, 0.5),
        (0.2,) * 5,
        (-0.1, 0.4, 0.4, 0.3),
        (-0.1, 0.6, 0.5),
        (0, 0, 0, 0),
        (0.3, 0.3, 0.3, 0.3),
        (float("nan"), 0, 0, 1),
        (float("inf"), 0, 0, 0),
    ],
)
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError):
        replace_random_underscore("_", probabilities)


def test_weighted_choice_supports_four_choices():
    assert weighted_choice(["xor", "parallel", "sequence", "loop"], LOOP_ONLY) == "loop"


@pytest.mark.parametrize(
    "expression",
    [
        "(! _)",
        "(! (_ || T1))",
        "(! (T1 , _))",
        "(! ((T1 ^ T2) || _))",
        "(! ((T1 ^ T2) , _))",
        "(! (T1 ^ (! (_ || T2))))",
    ],
)
def test_loop_only_generation_stops_without_an_eligible_position(expression):
    assert replace_random_underscore(expression, LOOP_ONLY) == expression


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(! (T1 ^ _))", "(! (T1 ^ (! _)))"),
        ("(! (T1 || (T2 ^ _)))", "(! (T1 || (T2 ^ (! _))))"),
        ("(! ((T1 ^ _) , T2))", "(! ((T1 ^ (! _)) , T2))"),
        ("(! (T1 ^[p] _))", "(! (T1 ^[p] (! _)))"),
        ("(! (T1 ^ (! (T2 ^ _))))", "(! (T1 ^ (! (T2 ^ (! _)))))"),
        ("((! _) || _)", "((! _) || (! _))"),
        ("(! (_ || (T1 ^ _)))", "(! (_ || (T1 ^ (! _))))"),
    ],
)
def test_loop_only_generation_uses_a_valid_ancestor_path(expression, expected):
    result = replace_random_underscore(expression, LOOP_ONLY)
    assert result == expected
    assert_valid_loop_ancestry(replace_underscores(result))


@pytest.mark.parametrize(
    "expression",
    [
        "(! (! _))",
        "(! (T1 || (! _)))",
        "(! (T1 , (! _)))",
        "(! ((T1 ^ T2) || (! _)))",
        "(! (T1 ^ (! (! _))))",
        "((! (! T1)) || _)",
    ],
)
def test_invalid_existing_loop_nesting_is_rejected(expression):
    with pytest.raises(ValueError):
        replace_random_underscore(expression)


@pytest.mark.parametrize("expression", ["T1", "(! T1)", "(! (T1 ^ (! T2)))"])
def test_finished_process_is_unchanged(expression):
    assert replace_random_underscore(expression, LOOP_ONLY) == expression


@pytest.mark.parametrize("seed", [0, 7, 31, 99, 2026])
def test_repeated_random_expansion_preserves_loop_constraint(seed):
    random.seed(seed)
    expression = "_"
    for _ in range(60):
        expression = replace_random_underscore(expression, (0.3, 0.2, 0.2, 0.3))
        named_expression = replace_underscores(expression)
        assert_valid_loop_ancestry(named_expression)
    tree = PARSER.parse(named_expression)
    tasks = get_tasks(tree)
    assert tasks == {f"T{index}" for index in range(1, len(tasks) + 1)}
    assert "_" not in named_expression
    assert assert_valid_loop_ancestry(named_expression) > 0


@pytest.mark.parametrize(
    ("expression", "nested", "independent", "pareto"),
    [
        ("T1", 0, 0, 1),
        ("(! T1)", 1, 1, 1),
        ("(! (T1 ^ T2))", 2, 1, 2),
        ("((! T1) || (! T2))", 1, 2, 1),
        ("((! T1), (! T2))", 1, 2, 1),
        ("(! ((T1 ^ T2) || (T3 ^ (T4 ^ T5))))", 3, 2, 6),
        ("((! (T1 ^ T2)), (! (T3 ^ T4)))", 2, 2, 4),
        ("(! (T1 ^ (! (T2 ^ T3))))", 4, 1, 3),
    ],
)
def test_loops_count_like_xor_in_complexity_metrics(expression, nested, independent, pareto):
    assert max_nested_xor(expression) == nested
    assert max_independent_xor(expression) == independent
    assert max_theoretical_pareto_length(PARSER.parse(expression)) == pareto


def test_batch_generation_produces_ten_distinct_valid_processes():
    random.seed(2026)
    forbidden = {"(T1 ^ T2)"}
    processes = generate_multiple_processes(
        probabilities=(0.3, 0.2, 0.2, 0.3),
        target_max_nested_xor=1,
        target_max_independent_xor=1,
        number_of_replacements=30,
        forbidden_processes=forbidden,
        num_processes=10,
        num_trials=100,
    )
    assert len(processes) == 10
    assert processes.isdisjoint(forbidden)
    assert forbidden == {"(T1 ^ T2)"}
    assert any("!" in expression for expression in processes)
    for expression in processes:
        assert "_" not in expression
        assert_valid_loop_ancestry(expression)
        assert max_nested_xor(expression) == 1
        assert max_independent_xor(expression) == 1
