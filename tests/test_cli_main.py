"""Tests that drive cli.main(argv) directly (not via subprocess).

These pin the CLI correctness fixes: stdout is never a mix of report + model,
-o is always honored, and explicit -m/-s always beat a preset.
"""

import pytest

from arpabo.cli import main


@pytest.fixture
def corpus(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(
        "the cat sat on the mat\nthe dog sat on the log\na cat sat on a mat\nthe cat ran fast\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def testfile(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("the cat sat on the mat\n", encoding="utf-8")
    return p


def test_eval_without_output_keeps_stdout_clean(corpus, testfile, capsys):
    """`arpabo corpus --eval test > model.arpa` must yield a clean ARPA on stdout."""
    rc = main([str(corpus), "--eval", str(testfile)])
    captured = capsys.readouterr()
    assert rc == 0
    # Model on stdout, report on stderr -- never mixed.
    assert "\\data\\" in captured.out
    assert "Perplexity" not in captured.out
    assert "Perplexity Evaluation" in captured.err


def test_output_flag_honored_with_stats(corpus, tmp_path, capsys):
    """-o must write the model even in --stats mode (README advertises this)."""
    out = tmp_path / "model.arpa"
    rc = main([str(corpus), "-o", str(out), "--stats"])
    captured = capsys.readouterr()
    assert rc == 0
    assert out.exists()
    assert "\\data\\" in out.read_text(encoding="utf-8")
    # Stats are the deliverable of --stats: they go to stdout.
    assert "Model Statistics" in captured.out


def test_stats_without_output_writes_no_model(corpus, capsys):
    rc = main([str(corpus), "--stats"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Model Statistics" in captured.out
    assert "\\data\\" not in captured.out


def _model_max_order(path):
    orders = [int(line.split("=")[0].split()[1]) for line in path.read_text().splitlines() if line.startswith("ngram ")]
    return max(orders)


def test_explicit_order_beats_preset(corpus, tmp_path):
    out = tmp_path / "m.arpa"
    rc = main([str(corpus), "-o", str(out), "-m", "3", "--preset", "accurate"])
    assert rc == 0
    assert _model_max_order(out) == 3


def test_preset_applies_when_order_not_given(corpus, tmp_path):
    out = tmp_path / "m.arpa"
    rc = main([str(corpus), "-o", str(out), "--preset", "accurate"])
    assert rc == 0
    # "accurate" recommends a higher order than the default 3.
    assert _model_max_order(out) > 3


def test_plain_build_writes_model_to_stdout(corpus, capsys):
    rc = main([str(corpus)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "\\data\\" in captured.out
