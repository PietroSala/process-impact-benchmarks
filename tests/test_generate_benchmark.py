import random

import pytest

import generate_benchmark as benchmark
from random_diagram_generation import replace_random_underscore
from stats import max_independent_xor, max_nested_xor


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = random.getstate()
    yield
    random.setstate(state)


def test_tiny_grid_contains_valid_loops_and_resumes_untouched(tmp_path, monkeypatch):
    args = [
        "--output", str(tmp_path), "--max-nested", "2", "--max-independent", "2",
        "--processes", "2", "--trials", "300", "--replacements", "100", "--seed", "42",
    ]
    benchmark.main(args)
    expected = {
        f"generated_processes_full_{nested}_{independent}.txt"
        for nested in (1, 2) for independent in (1, 2)
    }
    expected.add("benchmark_manifest.json")
    assert {path.name for path in tmp_path.iterdir()} == expected
    all_processes = set()
    for nested in (1, 2):
        for independent in (1, 2):
            path = tmp_path / f"generated_processes_full_{nested}_{independent}.txt"
            processes = path.read_text(encoding="utf-8").splitlines()
            assert len(processes) == len(set(processes)) == 2
            for process in processes:
                assert "!" in process and "_" not in process
                assert replace_random_underscore(process) == process
                assert max_nested_xor(process) == nested
                assert max_independent_xor(process) == independent
            all_processes.update(processes)
    assert len(all_processes) == 8
    snapshot = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}

    def unexpected_generation(*args, **kwargs):
        pytest.fail("A complete benchmark must not generate variants on resume")

    monkeypatch.setattr(benchmark, "generate_process", unexpected_generation)
    benchmark.main(args)
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()} == snapshot


def test_exhausted_budget_preserves_partial_file(tmp_path, monkeypatch, capsys):
    path = tmp_path / "generated_processes_full_1_1.txt"
    saved = "(! T1)\n"
    results = iter(["(! T1)", None, None, None])
    monkeypatch.setattr(benchmark, "generate_process", lambda *args: next(results))
    with pytest.raises(SystemExit) as error:
        benchmark.main([
            "--output", str(tmp_path), "--max-nested", "1", "--max-independent", "1",
            "--processes", "2", "--trials", "3", "--replacements", "100",
        ])
    assert error.value.code == 1
    assert path.read_text(encoding="utf-8") == saved
    assert "Completed variants are saved" in capsys.readouterr().err


def test_resume_rejects_changed_sampling_settings(tmp_path):
    args = ["--output", str(tmp_path), "--max-nested", "1", "--max-independent", "1", "--processes", "1"]
    benchmark.main(args)
    path = tmp_path / "generated_processes_full_1_1.txt"
    previous = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        benchmark.main(args + ["--sampling", "rejection"])
    assert error.value.code == 1
    assert path.read_bytes() == previous


def test_old_xor_only_classification_is_rejected(tmp_path):
    path = tmp_path / "generated_processes_full_1_1.txt"
    path.write_text("(! (T1 ^ T2))\n", encoding="utf-8")
    with pytest.raises(ValueError, match="XOR \\+ loop counts"):
        benchmark.load_existing(path, 1, 1, 10)


@pytest.mark.parametrize("weights", [
    ["0.4", "0.2", "0.2", "0.3"], ["-0.1", "0.3", "0.3", "0.5"],
    ["nan", "0.2", "0.2", "0.2"], ["0.5", "0.25", "0.25", "0"],
    ["0", "0.4", "0.4", "0.2"], ["0.8", "0", "0", "0.2"],
])
def test_invalid_probabilities_are_rejected(tmp_path, weights):
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--output", str(tmp_path), "--probabilities", *weights])
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())
