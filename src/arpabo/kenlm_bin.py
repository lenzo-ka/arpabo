"""Read and write the PocketSphinx / KenLM "Trie Language Model" binary format.

PocketSphinx stores n-gram LMs as a KenLM-style bit-packed reverse trie (header
``"Trie Language Model"``). Its own ``pocketsphinx_lm_convert`` and the Python
``NGramModel.write`` both crash converting *this* binary back to ARPA (a bug in the
C ``ngram_model_trie_write_arpa`` trie-count assertion), so this module reads the
binary directly and lets arpabo emit ARPA — and writes the binary natively too, so
arpabo can convert ARPA <-> binary in both directions without the buggy tool.

The layout is reconstructed from the PocketSphinx C sources (``src/lm/``:
``ngram_model_trie.c``, ``lm_trie.c``, ``lm_trie_quant.c``, ``bitarr.c``). See
``docs/kenlm-binary-format.md`` for the full byte/bit spec. Little-endian throughout.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from math import log10

TRIE_HDR = b"Trie Language Model"          # 19 bytes, not NUL-terminated
_DEFAULT_LOGBASE = 1.0001                    # pocketsphinx logmath default (shift 0)
_PROB_BITS = 16                              # fixed 16-bit quantized prob/backoff tables
_BO_BITS = 16
_MID_TABLE = (1 << _PROB_BITS) + (1 << _BO_BITS)   # 131072 floats per middle order
_LONG_TABLE = 1 << _PROB_BITS                       # 65536 floats for the longest order


@dataclass
class LMData:
    """A decoded n-gram LM: log10 probs + back-offs, order-indexed (order 1 = index 0)."""

    order: int
    vocab: list[str]
    # ngrams[o] = list of (word_tuple, log10_prob, log10_backoff_or_None), o = 0..order-1
    ngrams: list[list[tuple[tuple[str, ...], float, float | None]]] = field(default_factory=list)

    @property
    def counts(self) -> list[int]:
        return [len(self.ngrams[o]) for o in range(self.order)]


def _required_bits(v: int) -> int:
    """Bits to store 0..v (bitarr_required_bits): floor(log2 v)+1, or 0 for v==0."""
    r = 0
    while v:
        v >>= 1
        r += 1
    return r


def _read_bits(mem: bytes, bit_offset: int, width: int) -> int:
    """Read `width` bits (<=57) at `bit_offset` — little-endian 8-byte window, shift, mask."""
    if width == 0:
        return 0
    byte = bit_offset >> 3
    shift = bit_offset & 7
    chunk = int.from_bytes(mem[byte:byte + 8].ljust(8, b"\x00"), "little")
    return (chunk >> shift) & ((1 << width) - 1)


def read_kenlm_bin(path: str, logbase: float = _DEFAULT_LOGBASE) -> LMData:
    """Read a PocketSphinx trie-binary LM into an :class:`LMData` (log10 values).

    `logbase` is the logmath base the file was built with (pocketsphinx default 1.0001);
    stored logmath-internal floats are scaled to log10 by ``* log10(logbase)``.
    """
    scale = log10(logbase)
    with open(path, "rb") as fh:
        data = fh.read()
    pos = 0

    hdr = data[pos:pos + len(TRIE_HDR)]
    pos += len(TRIE_HDR)
    if hdr != TRIE_HDR:
        raise ValueError(f"not a PocketSphinx trie LM (header={hdr!r})")
    order = data[pos]
    pos += 1
    counts = list(struct.unpack_from(f"<{order}I", data, pos))
    pos += 4 * order

    # ---- quantization tables (order > 1): 4-byte dummy, then float32 tables ----
    quant: tuple[float, ...] = ()
    if order > 1:
        pos += 4  # discard the (always-1) quant-type dummy
        nvalues = (order - 2) * _MID_TABLE + _LONG_TABLE
        quant = struct.unpack_from(f"<{nvalues}f", data, pos)
        pos += 4 * nvalues

    # ---- unigram array: (counts[0]+1) x {float prob; float bo; uint32 next} ----
    n_ug = counts[0]
    ug = [struct.unpack_from("<ffI", data, pos + 12 * i) for i in range(n_ug + 1)]
    pos += 12 * (n_ug + 1)

    # ---- ngram_mem: bit-packed middle arrays (orders 2..order-1) then longest ----
    word_bits = _required_bits(counts[0])
    mids = []          # per middle order: byte offset, total_bits, next_bits, quant table base
    cur = 0
    for m in range(2, order):                       # middle order m -> array index m-2
        entries = counts[m - 1]
        next_bits = _required_bits(counts[m])
        total_bits = word_bits + _PROB_BITS + _BO_BITS + next_bits
        size = ((1 + entries) * total_bits + 7) // 8 + 8
        mids.append({"off": cur, "total_bits": total_bits, "next_bits": next_bits,
                     "qbase": (m - 2) * _MID_TABLE})
        cur += size
    long_off = long_total = 0
    if order > 1:
        l_entries = counts[order - 1]
        long_total = word_bits + _PROB_BITS
        long_off = cur
        cur += ((1 + l_entries) * long_total + 7) // 8 + 8
    ngram_mem = data[pos:pos + cur]
    pos += cur

    # ---- word strings: uint32 total bytes, then that many NUL-terminated ASCII words ----
    (kbytes,) = struct.unpack_from("<I", data, pos)
    pos += 4
    block = data[pos:pos + kbytes]
    vocab = [w.decode("latin-1") for w in block.split(b"\x00")[:n_ug]]

    long_qbase = (order - 2) * _MID_TABLE

    def mid_entry(mi: int, ptr: int):
        m = mids[mi]
        b = m["off"] * 8 + ptr * m["total_bits"]
        wid = _read_bits(ngram_mem, b, word_bits)
        bo_i = _read_bits(ngram_mem, b + word_bits, _BO_BITS)
        p_i = _read_bits(ngram_mem, b + word_bits + _BO_BITS, _PROB_BITS)
        nxt = _read_bits(ngram_mem, b + word_bits + _BO_BITS + _PROB_BITS, m["next_bits"])
        return wid, quant[m["qbase"] + p_i] * scale, quant[m["qbase"] + (1 << _PROB_BITS) + bo_i] * scale, nxt

    def long_entry(ptr: int):
        b = long_off * 8 + ptr * long_total
        wid = _read_bits(ngram_mem, b, word_bits)
        p_i = _read_bits(ngram_mem, b + word_bits, _PROB_BITS)
        return wid, quant[long_qbase + p_i] * scale

    out: list[list] = [[] for _ in range(order)]
    # 1-grams straight from the unigram array (raw floats scaled to log10).
    for w in range(n_ug):
        out[0].append(((vocab[w],), ug[w][0] * scale, ug[w][1] * scale if order > 1 else None))

    # descend the reverse trie: root unigram = LAST word; prepend earlier context words.
    def descend(path: list[int], begin: int, end: int, depth: int) -> None:
        if depth == order - 1:                       # children are longest (top-order leaves)
            for ptr in range(begin, end):
                wid, lp = long_entry(ptr)
                words = tuple(vocab[x] for x in reversed(path + [wid]))
                out[order - 1].append((words, lp, None))
        else:
            mi = depth - 1
            for ptr in range(begin, end):
                wid, lp, lbo, cbeg = mid_entry(mi, ptr)
                _, _, _, cend = mid_entry(mi, ptr + 1)   # end = next slot's next-pointer
                words = tuple(vocab[x] for x in reversed(path + [wid]))
                out[depth].append((words, lp, lbo))
                descend(path + [wid], cbeg, cend, depth + 1)

    if order > 1:
        for w in range(n_ug):
            descend([w], ug[w][2], ug[w + 1][2], 1)

    return LMData(order=order, vocab=vocab, ngrams=out)


def write_arpa(lm: LMData, out) -> None:
    """Write `lm` as ARPA text to a file object (values are already log10)."""
    out.write("\\data\\\n")
    for o in range(lm.order):
        out.write(f"ngram {o + 1}={len(lm.ngrams[o])}\n")
    out.write("\n")
    for o in range(lm.order):
        out.write(f"\\{o + 1}-grams:\n")
        for words, lp, bo in lm.ngrams[o]:
            ws = " ".join(words)
            if bo is None:
                out.write(f"{lp:.4f} {ws}\n")
            else:
                out.write(f"{lp:.4f} {ws} {bo:.4f}\n")
        out.write("\n")
    out.write("\\end\\\n")


def kenlm_bin_to_arpa(bin_path: str, arpa_path: str, logbase: float = _DEFAULT_LOGBASE) -> LMData:
    """Convert a PocketSphinx trie-binary LM to an ARPA file. Returns the decoded LMData."""
    lm = read_kenlm_bin(bin_path, logbase=logbase)
    with open(arpa_path, "w") as fh:
        write_arpa(lm, fh)
    return lm
