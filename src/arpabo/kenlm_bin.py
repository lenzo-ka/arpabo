"""Read and write PocketSphinx's binary n-gram LM format ("Trie Language Model").

IMPORTANT — this is the **PocketSphinx-SPECIFIC** binary, not the generic/upstream
KenLM binary. It is a KenLM-*derived* bit-packed reverse trie (header
``"Trie Language Model"``), but it is **not byte-compatible with upstream KenLM's
``build_binary`` output**: PocketSphinx uses its own header, always-on 16-bit
quantization, logmath-internal value scaling (log10 x 1/log10(1.0001)), and its own
vocab layout. Upstream KenLM uses an ``"mmap lm ..."`` header, PROBING or TRIE
structures, optional quantization + Bhiksha compression, and direct log10 values.
The portable interchange between the two (and every other tool) is **ARPA text** —
which arpabo reads/writes elsewhere; this module is only the sphinx-binary bridge.

Why it exists: PocketSphinx's own bin->ARPA path (``pocketsphinx_lm_convert``,
``NGramModel.write``) crashes on some of its own binaries (a bug in the C
``ngram_model_trie_write_arpa`` trie-count assertion). This module reads the sphinx
binary directly (so arpabo can emit ARPA from it) and writes it natively, so arpabo
converts ARPA <-> sphinx-binary both ways without the buggy tool.

The layout is reconstructed from the PocketSphinx C sources (``src/lm/``:
``ngram_model_trie.c``, ``lm_trie.c``, ``lm_trie_quant.c``, ``bitarr.c``). See
``docs/sphinx-lm-binary.md`` for the full byte/bit spec. Little-endian throughout.
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


def read_arpa(path: str) -> LMData:
    """Parse an ARPA file into an :class:`LMData` (log10 values; word ids = 1-gram file order)."""
    order = 0
    ngrams: dict[int, list] = {}
    cur = 0
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("ngram ") and "=" in s:
                order = max(order, int(s.split("=")[0].split()[1]))
            elif s.startswith("\\") and s.endswith("-grams:"):
                cur = int(s[1:].split("-")[0])
                ngrams[cur] = []
            elif cur and s and (s[0].isdigit() or s[0] == "-"):
                p = s.split()
                try:
                    lp = float(p[0])
                except ValueError:
                    continue
                words = tuple(p[1:1 + cur])
                bo = None
                if len(p) == cur + 2:
                    try:
                        bo = float(p[cur + 1])
                    except ValueError:
                        bo = None
                ngrams[cur].append((words, lp, bo))
    ng = [ngrams.get(o, []) for o in range(1, order + 1)]
    return LMData(order=order, vocab=[w[0] for w in ng[0]], ngrams=ng)


def arpa_to_kenlm_bin(arpa_path: str, bin_path: str, logbase: float = _DEFAULT_LOGBASE) -> LMData:
    """Convert an ARPA file to a PocketSphinx trie-binary LM (native — no external tool)."""
    lm = read_arpa(arpa_path)
    write_kenlm_bin(lm, bin_path, logbase=logbase)
    return lm


# ---------------------------------------------------------------------------
# Writer: LMData -> PocketSphinx trie binary (the inverse of read_kenlm_bin)
# ---------------------------------------------------------------------------

import bisect  # noqa: E402


def _make_bins(values: list[float], bins: int = 1 << _PROB_BITS) -> list[float]:
    """Train a 65536-entry quantization table (lm_trie_quant make_bins): equal-COUNT bins over the sorted
    values, each bin's center = the mean of its members (empty leading bin -> -inf, later empties repeat the
    previous center). For <65536 values each value lands in its own bin, so the table is LOSSLESS."""
    vs = sorted(values)
    n = len(vs)
    centers: list[float] = []
    start = 0
    prev = float("-inf")
    for i in range(bins):
        finish = n * (i + 1) // bins
        if finish == start:
            centers.append(prev if i else float("-inf"))
        else:
            c = sum(vs[start:finish]) / (finish - start)
            centers.append(c)
            prev = c
        start = finish
    return centers


def _encode(centers: list[float], value: float) -> int:
    """value -> nearest-center bin index (lm_trie_quant bins_encode; tie -> upper)."""
    idx = bisect.bisect_left(centers, value)
    if idx == 0:
        return 0
    if idx >= len(centers):
        return len(centers) - 1
    return idx - 1 if (value - centers[idx - 1]) < (centers[idx] - value) else idx


def _write_bits(buf: bytearray, bit_offset: int, width: int, value: int) -> None:
    """OR `value` (`width` bits) into the little-endian bit array at `bit_offset` (buf must be pre-zeroed)."""
    if width == 0:
        return
    byte = bit_offset >> 3
    shift = bit_offset & 7
    chunk = int.from_bytes(buf[byte:byte + 8], "little")
    chunk |= (value & ((1 << width) - 1)) << shift
    buf[byte:byte + 8] = chunk.to_bytes(8, "little")


def write_kenlm_bin(lm: LMData, path: str, logbase: float = _DEFAULT_LOGBASE) -> None:
    """Write `lm` as a PocketSphinx trie-binary LM (the inverse of :func:`read_kenlm_bin`).

    Reproduces the C build (``lm_trie_build``): word ids = 1-gram order; n-grams stored reversed and sorted
    per order; next-pointers = contiguous child ranges; 16-bit binned prob/backoff quant; raw-float unigrams.
    Handles the complete-context case (every k-gram's (k-1)-gram prefix present) — true for full ARPA LMs;
    it raises on a missing intermediate context (a pruned LM needing synthesized 'dummy' entries — a TODO).
    Not byte-identical to PocketSphinx's own output (that needs its truncating sort comparator), but a valid,
    round-trippable, PocketSphinx-loadable binary.
    """
    inv = 1.0 / log10(logbase)                  # log10 value -> stored logmath-internal float
    order = lm.order
    vocab = [w[0] for (w, _p, _b) in lm.ngrams[0]]
    wid = {w: i for i, w in enumerate(vocab)}
    n_ug = len(vocab)

    def rq(x):                                  # requantize None-safe log10 -> stored float
        return 0.0 if x is None else x * inv

    # reversed, id-keyed, sorted entries per order o (2..order): (rev_ids, prob_stored, bo_stored)
    arr: dict[int, list] = {}
    for o in range(2, order + 1):
        es = [(tuple(wid[w] for w in reversed(words)), lp * inv, rq(bo))
              for (words, lp, bo) in lm.ngrams[o - 1]]
        es.sort(key=lambda e: e[0])
        arr[o] = es
    out_counts = [n_ug] + [len(arr[o]) for o in range(2, order + 1)]

    word_bits = _required_bits(n_ug)

    def child_next(parent_entries, child_entries, plen):
        """next[i] = begin index of parent i's children (child prefix rev_ids[:plen]); + sentinel end."""
        from collections import Counter
        cc = Counter(e[0][:plen] for e in child_entries)
        pset = {pe[0] for pe in parent_entries}
        for pref in cc:
            if pref not in pset:
                raise NotImplementedError(
                    "pruned LM: missing intermediate context (dummy entries) not yet supported")
        nxt, cum = [], 0
        for pe in parent_entries:
            nxt.append(cum)
            cum += cc.get(pe[0], 0)
        nxt.append(cum)
        return nxt

    # ---- unigram next pointers (children = order-2 entries grouped by rev_ids[0]) ----
    from collections import Counter
    c2 = Counter(e[0][0] for e in arr.get(2, []))
    ug_next, cum = [], 0
    for w in range(n_ug):
        ug_next.append(cum)
        cum += c2.get(w, 0)
    ug_next.append(cum)

    # ---- quant tables + per-entry indices ----
    values: list[float] = []
    mid_tables = {}       # order -> (prob_centers, bo_centers)
    for m in range(2, order):
        pc = _make_bins([e[1] for e in arr[m]])
        bc = _make_bins([e[2] for e in arr[m]])
        mid_tables[m] = (pc, bc)
        values.extend(pc)
        values.extend(bc)
    long_centers = _make_bins([e[1] for e in arr[order]]) if order > 1 else []
    values.extend(long_centers)

    # ---- bit-pack each order's array ----
    ngram_mem = bytearray()
    for m in range(2, order):                          # middle order m -> m-grams, next into (m+1)-grams
        entries = arr[m]
        next_bits = _required_bits(out_counts[m])
        total_bits = word_bits + 2 * _PROB_BITS + next_bits
        nxt = child_next(entries, arr[m + 1], m)
        buf = bytearray(((1 + len(entries)) * total_bits + 7) // 8 + 8)
        pc, bc = mid_tables[m]
        for i, (rev, ps, bs) in enumerate(entries):
            b = i * total_bits
            _write_bits(buf, b, word_bits, rev[-1])                       # the new (deepest) word
            _write_bits(buf, b + word_bits, _BO_BITS, _encode(bc, bs))
            _write_bits(buf, b + word_bits + _BO_BITS, _PROB_BITS, _encode(pc, ps))
            _write_bits(buf, b + word_bits + 2 * _PROB_BITS, next_bits, nxt[i])
        _write_bits(buf, len(entries) * total_bits + word_bits + 2 * _PROB_BITS, next_bits, nxt[-1])  # sentinel
        ngram_mem += buf
    if order > 1:                                      # longest order (top): word + prob, no next
        entries = arr[order]
        total_bits = word_bits + _PROB_BITS
        buf = bytearray(((1 + len(entries)) * total_bits + 7) // 8 + 8)
        for i, (rev, ps, _bs) in enumerate(entries):
            b = i * total_bits
            _write_bits(buf, b, word_bits, rev[-1])
            _write_bits(buf, b + word_bits, _PROB_BITS, _encode(long_centers, ps))
        ngram_mem += buf

    # ---- assemble the file ----
    out = bytearray()
    out += TRIE_HDR
    out += bytes([order])
    out += struct.pack(f"<{order}I", *out_counts)
    if order > 1:
        out += struct.pack("<i", 1)                    # quant-type dummy
        out += struct.pack(f"<{len(values)}f", *values)
    probs0 = {w[0]: (lp, bo) for (w, lp, bo) in lm.ngrams[0]}
    for i in range(n_ug):
        lp, bo = probs0[vocab[i]]
        out += struct.pack("<ffI", lp * inv, rq(bo), ug_next[i])
    out += struct.pack("<ffI", 0.0, 0.0, ug_next[n_ug])   # sentinel unigram
    out += ngram_mem
    blob = b"".join(w.encode("latin-1") + b"\x00" for w in vocab)
    out += struct.pack("<I", len(blob)) + blob

    with open(path, "wb") as fh:
        fh.write(out)
