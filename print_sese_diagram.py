"""Render a process with its XOR + loop complexity measures.

Examples (run from the project directory):
    python print_sese_diagram.py --expression "(! (T1 ^ T2))"
    python print_sese_diagram.py --file generated_processes_loops_regenerated/generated_processes_full_1_4.txt
"""

import argparse
from pathlib import Path

# Share the parser and drawing code with the notebook and generator. The old
# copy imported a nonexistent env module and rendered XORs as parallel gateways.
from sese_diagram import (
    PARSER as SESE_PARSER,
    PATH_IMAGE_BPMN_LARK,
    PATH_IMAGE_BPMN_LARK_SVG,
    RESOLUTION,
    dot_sese_diagram,
    dot_task,
    dot_exclusive_gateway,
    dot_probabilistic_gateway,
    dot_loop_gateway,
    dot_parallel_gateway,
    dot_rectangle_node,
    get_tasks,
    print_sese_diagram,
    wrap_sese_diagram,
)
from stats import max_independent_xor, max_nested_xor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--expression", help="process expression; default: (! (T1 ^ T2))")
    source.add_argument("--file", type=Path, help="benchmark text file, one process per line")
    parser.add_argument("--process-number", type=int, default=1, help="one-based line number in --file")
    parser.add_argument("--output", type=Path, default=Path("bpmn_preview"), help="output path stem for PNG and SVG")
    args = parser.parse_args(argv)
    if args.process_number < 1:
        parser.error("--process-number must be at least 1")
    expression = args.expression or "(! (T1 ^ T2))"
    if args.file:
        processes = args.file.read_text(encoding="utf-8").splitlines()
        if args.process_number > len(processes):
            parser.error(f"{args.file} contains only {len(processes)} processes")
        expression = processes[args.process_number - 1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    svg_path = args.output.with_suffix(".svg")
    diagram = print_sese_diagram(expression, outfile=str(png_path), outfile_svg=str(svg_path))
    diagram.close()
    print(f"Nesting (XOR + loop): {max_nested_xor(expression)}")
    print(f"Independent (XOR + loop): {max_independent_xor(expression)}")
    print(f"PNG: {png_path.resolve()}")
    print(f"SVG: {svg_path.resolve()}")


if __name__ == "__main__":
    main()
