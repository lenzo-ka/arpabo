"""Tests for the PocketSphinx / KenLM trie-binary reader (and writer, once present).

The fixed 65536-entry quant tables make even a tiny .lm.bin ~786 KB, so we do NOT
commit a binary fixture. Instead:
  * if pocketsphinx_lm_convert is available, convert the tiny ARPA -> bin and check
    that the native reader reproduces the ARPA (all orders) — else skip.
  * a pure write->read round-trip runs whenever arpabo can write the binary itself
    (no external tool), exercising the reader against arpabo's own writer.
"""

import os
import shutil
import subprocess

import pytest

from arpabo.kenlm_bin import read_kenlm_bin, write_arpa

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "tiny.arpa")


def _find_converter():
    p = shutil.which("pocketsphinx_lm_convert")
    if p:
        return p
    cand = os.path.expanduser("~/dev/cmu/pocketsphinx/build/pocketsphinx_lm_convert")
    return cand if os.path.exists(cand) else None


def _parse_arpa(path):
    """{order: {word_tuple: log10_prob}} from an ARPA file."""
    out, cur = {}, 0
    for line in open(path):
        s = line.strip()
        if s.startswith("\\") and s.endswith("-grams:"):
            cur = int(s[1:].split("-")[0])
            out[cur] = {}
        elif cur and s and (s[0].isdigit() or s[0] == "-"):
            p = s.split()
            try:
                lp = float(p[0])
            except ValueError:
                continue
            # trailing back-off field (if any) is not a word; words = the `cur` tokens after prob
            out[cur][tuple(p[1 : 1 + cur])] = lp
    return out


@pytest.mark.skipif(_find_converter() is None, reason="pocketsphinx_lm_convert not available")
def test_read_bin_matches_arpa(tmp_path):
    """Convert the tiny ARPA to a trie binary, read it back, and compare every n-gram."""
    conv = _find_converter()
    binp = str(tmp_path / "tiny.lm.bin")
    subprocess.run([conv, "-i", FIX, "-o", binp], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lm = read_kenlm_bin(binp)
    orig = _parse_arpa(FIX)
    assert lm.order == max(orig)
    for o in range(lm.order):
        mine = {w: lp for (w, lp, _bo) in lm.ngrams[o]}
        want = orig[o + 1]
        assert set(mine) == set(want), f"order {o+1} n-gram set differs"
        for w in want:
            assert abs(mine[w] - want[w]) < 1e-3, f"order {o+1} {w}: {mine[w]} vs {want[w]}"
