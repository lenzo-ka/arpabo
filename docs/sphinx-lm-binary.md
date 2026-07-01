# PocketSphinx binary LM support (`arpabo.kenlm_bin`)

arpabo can read and write **PocketSphinx's binary n-gram LM** — the format whose files carry the header
`"Trie Language Model"` (e.g. `pocketsphinx/model/en-us/en-us.lm.bin`).

> **This is the PocketSphinx-SPECIFIC binary, not generic/upstream KenLM.**
> It is a KenLM-*derived* bit-packed reverse trie, but it is **not byte-compatible** with upstream KenLM's
> `build_binary` output. The portable interchange between tools is **ARPA text** (which arpabo reads/writes
> in the rest of the package). This module is only the bridge to/from *PocketSphinx's* binary.

## Why arpabo has its own reader/writer

PocketSphinx's own binary→ARPA path is **broken for some of its own models**: both `pocketsphinx_lm_convert`
and the Python `NGramModel.write` abort at an assertion in the C `ngram_model_trie_write_arpa`
(`raw_ngram_idx == n_counts[i-1]`) — a header/actual n-gram **count mismatch**. For `en-us.lm.bin` the
header claims 2,051,547 bigrams but only 2,051,541 are enumerable (the extra 6 are the synthesized
"dummy" context entries `lm_trie_fix_counts` adds; the writer miscounts them). So that LM could not be
converted back to ARPA by any pocketsphinx tool.

arpabo's reader parses the trie **directly** and enumerates the *actual* n-grams, so it does not crash on
that mismatch — it produces the full ARPA nothing else could.

## API

```python
from arpabo.kenlm_bin import (
    read_kenlm_bin, write_kenlm_bin,   # sphinx binary  <-> LMData
    read_arpa, write_arpa,             # ARPA           <-> LMData
    kenlm_bin_to_arpa, arpa_to_kenlm_bin,   # file-to-file convenience
    LMData,                            # order, vocab, ngrams[o] = [(words, log10_prob, log10_backoff|None)]
)
```

`arpabo.convert` mirrors the external-tool helpers with native ones:
`from_pocketsphinx_binary(bin, arpa)` (bin→ARPA) and `to_pocketsphinx_binary_native(arpa, bin)` (ARPA→bin,
no external tool). The CLI adds `--from-bin LM_BIN` (→ ARPA on `-o`/stdout); `--to-bin` now prefers the
native writer and falls back to `pocketsphinx_lm_convert` only for pruned LMs (below).

```
arpabo --from-bin en-us.lm.bin -o en-us.arpa      # sphinx binary -> ARPA (native)
arpabo corpus.txt -o model.arpa --to-bin          # build ARPA, then ARPA -> sphinx binary (native)
```

## Verification

- **Exact round-trip**: `slt.lm` ARPA → bin (pocketsphinx tool) → `read_kenlm_bin` reproduces all three
  orders exactly (2685 / 7976 / 9224, max |Δlog10| = 0.0000).
- **Exact unigrams** of `en-us.lm.bin` match pocketsphinx's own (crashing) ARPA output to 5 decimals.
- **Write round-trip**: `write_kenlm_bin` then `read_kenlm_bin` reproduces the LM exactly (self-contained,
  no tool), and **pocketsphinx loads arpabo's written binary** (`NGramModel`). See `tests/test_kenlm_bin.py`.

## Format (summary; full bit-level spec in the source docstrings)

Little-endian throughout. `header "Trie Language Model" (19B)`, `uint8 order`, `order× uint32 counts`,
`quant block` (a 4-byte flag + 16-bit binned prob/backoff float tables), `unigram array`
`(counts[0]+1)×{float32 prob; float32 bo; uint32 next}` (raw, un-quantized), the bit-packed `ngram_mem`
(middle orders then longest; entry = `[word_id | backoff_idx16 | prob_idx16 | next_ptr]`, longest =
`[word_id | prob_idx16]`), then `word strings` (`uint32 nbytes` + NUL-terminated ASCII).

Key points a naive reader/writer gets wrong, handled here:
- **Reverse/suffix trie**: the root unigram is the *last* word; descending prepends earlier context. n-grams
  are reconstructed by reversing the descent path.
- **Value scale**: stored floats are logmath-internal; `log10 = stored × log10(1.0001)`.
- **Quantization** is equal-count binning with mean centers; for < 65536 n-grams per order it is *lossless*
  (each value gets its own bin) — which is why small LMs round-trip exactly.

## Limitations / notes

- The **writer** supports **complete-context** LMs (every k-gram's (k-1)-gram prefix present) — true for full
  ARPA LMs, including everything arpabo produces. For a **pruned** LM (missing intermediate contexts, needing
  synthesized "dummy" entries), the writer raises `NotImplementedError`; use the pocketsphinx tool
  (`to_pocketsphinx_binary`) for those, or extend the writer with the `lm_trie_fix_counts` dummy pass.
- Output is a **valid, pocketsphinx-loadable** binary but **not byte-identical** to pocketsphinx's own output
  (that would require replicating its truncating `make_bins` sort comparator); the decoded values are
  equivalent.
- **Upstream KenLM** compatibility (the `"mmap lm ..."` format, PROBING/TRIE, Bhiksha, direct-log10) would be
  a **separate dialect**, not covered here. ARPA remains the cross-tool interchange.
