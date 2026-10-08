import random
import math

from lark import Tree
from sese_diagram import PARSER

def count_underscores(string):
    return string.count('_')

def replace_underscores(input_string):
    count = 0
    result = ""
    for char in input_string:
        if char == '_':
            count += 1
            result += f"T{count}"
        else:
            result += char
    return result

SEED_STRING = '_'
REPLACEMENTS = ("(_ ^ _)", "(_ || _)", "(_ , _)", "(! _)")
DEFAULT_PROBABILITIES = (0.3, 0.2, 0.2, 0.3)

def guess_three_numbers():
    # Generate two random numbers between 0 and 1
    a = random.random()
    b = random.random()

    # Ensure a <= b
    if a > b:
        a, b = b, a

    # Calculate the three numbers
    x = a
    y = b - a
    z = 1 - b

    return x, y, z

def guess_four_numbers():
    """Return probabilities for XOR, parallel, sequence, and loop."""
    boundaries = [0.0, *sorted(random.random() for _ in range(3)), 1.0]
    return tuple(right - left for left, right in zip(boundaries, boundaries[1:]))

def _validate_probabilities(choices, probabilities):
    if not choices or len(choices) != len(probabilities):
        raise ValueError("Must provide one probability per choice")
    if any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("Probabilities must be finite and nonnegative")
    if not abs(sum(probabilities) - 1.0) < 1e-6:
        raise ValueError("Probabilities must sum to 1")

def weighted_choice(choices, probabilities):
    _validate_probabilities(choices, probabilities)
    r = random.random() * sum(probabilities)
    cumulative = 0.0
    last_positive_choice = None
    for choice, probability in zip(choices, probabilities):
        if probability > 0:
            last_positive_choice = choice
        cumulative += probability
        if r < cumulative:
            return choice
    return last_positive_choice  # Guard against floating-point rounding.

def _replacement_positions(input_string):
    """Yield (placeholder offset, loop allowed) along each ancestor path.

    An XOR between two loops permits nesting. Parallel and sequential
    regions preserve the restriction imposed by the nearest loop.
    """
    def visit(node, loop_blocked=False):
        if not isinstance(node, Tree):
            return
        if node.data == 'loop':
            if loop_blocked:
                raise ValueError("Nested loops require an intervening XOR split")
            loop_blocked = True
        elif node.data in {'xor', 'xor_probability'}:
            loop_blocked = False
        elif node.data == 'task':
            token = node.children[0]
            if token.value == '_':
                yield token.start_pos, not loop_blocked
            return
        for child in node.children:
            yield from visit(child, loop_blocked)

    yield from visit(PARSER.parse(input_string))

def replace_random_underscore(input_string, probabilities= None):
    """Expand a task placeholder while preserving valid loop nesting.

    Four probabilities select XOR, parallel, sequence, and loop, respectively.
    Legacy three-probability calls keep generating loop-free processes. With
    no probabilities, the weights are (0.3, 0.2, 0.2, 0.3), respectively.
    Disallowed loops are excluded and the remaining weights renormalized.
    If no permitted expansion has positive weight, return the input unchanged.
    """
    if probabilities is None:
        probabilities = DEFAULT_PROBABILITIES
    else:
        probabilities = tuple(probabilities)
        if len(probabilities) == 3:
            probabilities += (0.0,)
    _validate_probabilities(REPLACEMENTS, probabilities)

    positions = [
        (position, loop_allowed)
        for position, loop_allowed in _replacement_positions(input_string)
        if loop_allowed or any(p > 0 for p in probabilities[:3])
    ]
    if not positions:
        return input_string

    random_position, loop_allowed = random.choice(positions)
    choices = REPLACEMENTS if loop_allowed else REPLACEMENTS[:3]
    weights = probabilities if loop_allowed else probabilities[:3]
    total_weight = sum(weights)
    random_replacement = weighted_choice(choices, [p / total_weight for p in weights])
    return input_string[:random_position] + random_replacement + input_string[random_position+1:]
