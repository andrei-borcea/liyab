#!/usr/bin/env python3
"""Generates tests/data/text_cases.txt, the expectations tests/test_text.cpp checks the regex engine, the
pre-tokenizer table and NFC / NFKC normalization against:

* regex splits: Python's `regex` module, cross-checked against the Hugging Face tokenizers' own Split
  (Oniguruma), which is what models are tokenized with. Where the two disagree (`^` and `$` match at line
  boundaries in Oniguruma, only at the text's ends in Python) the tokenizers' result is stored;
* pre-tokenizer splits: a tokenizers Sequence of Split pre-tokenizers built from PRE_TOKENIZERS, which mirrors
  the table in src/core/pre_tokenizer.cpp (the conformance runs of liyab-tok against each model's own
  tokenizer.json check that the table is right; this file checks that the engine runs it);
* NFC: Python's unicodedata; NFKC: the source tokenizers Unicode 9 normalizer, including every changed scalar.

    pip install regex tokenizers
    python3 tools/gen_text_tests.py > tests/data/text_cases.txt
    python3 tools/gen_text_tests.py --nfkc-only  # only the additive NFKC section

Format, one record per line, fields separated by tabs, text escaped as \\\\ \\t and \\u{HEX} (any non-printable or
non-ASCII code point):
    T id text                         a text the other records refer to
    P id pattern                      a pattern the R records refer to
    R pattern_id keep_gaps text_id spans  a regex split of the text: each piece as its end code-point offset,
                                      or start-end when it does not start where the piece before ended
    E pattern                         a pattern that must fail to compile
    S pre_name text_id spans          a pre-tokenizer split of the text (NFC-normalized first when the row says so)
    B pre_name text_id spans          a pre-tokenizer split in byte offsets (steps after ByteLevel can split UTF-8)
    N input expected                  NFC normalization
    K input expected                  NFKC normalization
"""
import argparse
import os
import random
import sys
import unicodedata

import regex
from tokenizers import Regex, normalizers, pre_tokenizers

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Mirrors src/core/pre_tokenizer.cpp: name -> (steps [(pattern, keep_gaps)], nfc).
GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
LLAMA3 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN2 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN35 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|"
          r"\s*[\r\n]+|\s+(?!\S)|\s+")
GPT4O = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
         r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
         r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
DIGITS3 = r"\p{N}{1,3}"
HAN_KANA = r"[\x{4E00}-\x{9FA5}\x{3040}-\x{309F}\x{30A0}-\x{30FF}]+"
DSV3_WORDS = (r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|"
              r" ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
DS_LETTERS = (
    r"\s?[A-Za-z\x{B5}\x{C0}-\x{D6}\x{D8}-\x{F6}\x{F8}-\x{1BA}\x{1BC}-\x{1BF}\x{1C4}-\x{293}\x{295}-\x{2AF}"
    r"\x{370}-\x{373}\x{376}\x{377}\x{37B}-\x{37D}\x{37F}\x{386}\x{388}-\x{38A}\x{38C}\x{38E}-\x{3A1}"
    r"\x{3A3}-\x{3F5}\x{3F7}-\x{481}\x{48A}-\x{52F}\x{531}-\x{556}\x{10A0}-\x{10C5}\x{13A0}-\x{13F5}"
    r"\x{13F8}-\x{13FD}\x{1C90}-\x{1CBA}\x{1CBD}-\x{1CBF}\x{1D00}-\x{1D2B}\x{1D6B}-\x{1D77}\x{1D79}-\x{1D9A}"
    r"\x{1E00}-\x{1F15}\x{1F18}-\x{1F1D}\x{1F20}-\x{1F45}\x{1F48}-\x{1F4D}\x{1F50}-\x{1F57}\x{1F59}\x{1F5B}"
    r"\x{1F5D}\x{1F5F}-\x{1F7D}\x{1F80}-\x{1FB4}\x{1FB6}-\x{1FBC}\x{1FBE}\x{1FC2}-\x{1FC4}\x{1FC6}-\x{1FCC}"
    r"\x{1FD0}-\x{1FD3}\x{1FD6}-\x{1FDB}\x{1FE0}-\x{1FEC}\x{1FF2}-\x{1FF4}\x{1FF6}-\x{1FFC}\x{2102}\x{2107}"
    r"\x{210A}-\x{2113}\x{2115}\x{2119}-\x{211D}\x{2124}\x{2126}\x{2128}\x{212A}-\x{212D}\x{212F}-\x{2134}"
    r"\x{2139}\x{213C}-\x{213F}\x{2145}-\x{2149}\x{214E}\x{2183}\x{2184}\x{2C00}-\x{2C7B}\x{2C7E}-\x{2CE4}"
    r"\x{2CEB}-\x{2CEE}\x{2CF2}\x{2CF3}\x{A640}-\x{A66D}\x{A680}-\x{A69B}\x{A722}-\x{A76F}\x{A771}-\x{A787}"
    r"\x{A78B}-\x{A78E}\x{AB70}-\x{ABBF}\x{FB00}-\x{FB06}\x{FB13}-\x{FB17}\x{FF21}-\x{FF3A}\x{FF41}-\x{FF5A}"
    r"\x{10400}-\x{1044F}\x{104B0}-\x{104D3}\x{104D8}-\x{104FB}\x{10C80}-\x{10CB2}\x{10CC0}-\x{10CF2}"
    r"\x{118A0}-\x{118DF}\x{1E900}-\x{1E943}]+")
DS_PUNCT = r"\s?[!-/:-~\x{FF01}-\x{FF0F}\x{FF1A}-\x{FF5E}\x{2018}-\x{201F}\x{3000}-\x{3002}]+"
DS_CJK = r"[\x{4E00}-\x{9FA5}\x{800}-\x{4E00}\x{AC00}-\x{D7FF}]+"
TEKKEN = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
          r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
          r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
# Youtu-LLM's \p{Hangul}, spelled out as the script's ranges like the engine's table.
YOUTU_CJK = (r"[\x{1100}-\x{11FF}\x{302E}\x{302F}\x{3131}-\x{318E}\x{3200}-\x{321E}\x{3260}-\x{327E}"
             r"\x{A960}-\x{A97C}\x{AC00}-\x{D7A3}\x{D7B0}-\x{D7C6}\x{D7CB}-\x{D7FB}\x{FFA0}-\x{FFBE}\x{FFC2}-\x{FFC7}"
             r"\x{FFCA}-\x{FFCF}\x{FFD2}-\x{FFD7}\x{FFDA}-\x{FFDC}]+|"
             r"[\x{FF01}\x{2026}\x{201C}\x{201D}\x{2018}\x{2019}\x{2014}\x{FF1A}\x{FF1B}\x{FF0C}\x{3001}-\x{303F}"
             r"\x{FE30}-\x{FE4F}]+|[\x{3105}-\x{312F}]+|[\x{4E00}-\x{9FA5}\x{3040}-\x{309F}\x{30A0}-\x{30FF}]+")
YOUTU_WORDS = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
               r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
               r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
PRE_TOKENIZERS = {
    "default": ([(GPT2, True)], False),
    "llama3": ([(LLAMA3, True)], False),
    "qwen2": ([(QWEN2, True)], True),
    "qwen35": ([(QWEN35, True)], True),
    "hunyuan": ([(QWEN2, True)], False),
    "deepseek-v3": ([(DIGITS3, True), (HAN_KANA, True), (DSV3_WORDS, True)], False),
    "deepseek-llm": ([(r"[\r\n]", True), (DS_LETTERS, True), (DS_PUNCT, True), (r"\s+$", True), (DS_CJK, True),
                      (r"\p{N}", True)], False),
    "gpt-4o": ([(GPT4O, False)], False),
    "smollm": ([(r"\p{N}", True), (GPT2, True)], False),
    "tekken": ([(TEKKEN, True)], False),
    "youtu": ([(YOUTU_CJK, True), (YOUTU_WORDS, True)], False),
}

# Engine features beyond the table's patterns: (pattern, keep_gaps, text).
FEATURES = [
    (r"a+?b", True, "aaab ab b"),
    (r"\d{2,3}?", True, "12345 6"),
    (r"\p{L}++a|\p{L}+", True, "aaa bab"),
    (r"(?>a+)b|a", True, "aaab aa"),
    (r"(?=ab)a|b", True, "ab b ba"),
    (r"x(?!y)", True, "xy xz x"),
    (r"^\s+|\s+$", True, "  a b  \n  c  "),
    (r"\A\s|\s\z", True, " a b "),
    (r"(?i:hello|'s)", True, "HeLLo 'S '\u017f 's hELLO"),
    (r"(?i)k+", True, "kKK\u212a k"),
    (r"(?:ab){2}|ab", True, "ababab"),
    (r"(?:ab){1,2}?c", True, "ababc abc"),
    (r"[^\s\p{L}]+", True, "a1!2 b"),
    (r"\P{L}+", True, "ab12 cd"),
    (r"\p{Lu}\p{Ll}*", True, "Hello World \u00c9COLE \u00c9cole"),
    (r"[\p{Lu}\p{Lt}]", True, "\u01c5x\u01c8"),
    (r"\w+", True, "h\u00e9llo_w\u00f6rld 123 a\u00b7b"),
    (r"\d+", True, "\u0663" + "4 \u00bd"),
    (r".+", True, "a\nb"),
    (r"[a-c\x{3B1}-\x{3B3}]+", True, "abc\u03b1\u03b2\u03b3d"),
    (r"\x41\u0042\x{43}", True, "xABCx"),
    (r"[\-\]\\]+", True, "a-]\\b"),
    (r"a{2", True, "a{2 a"),
    (r"\s+(?!\S)", True, "a   b    "),
    (r"[\r\n]+|.", True, "a\r\nb"),
    (r"\p{N}+", False, "ab12cd345"),
    (r"\p{Letter}+|\p{Nd}", True, "abc1"),
    (r"(?:a|ab)(?:c|bcd)", True, "abcd"),
    (r"[^a]", True, "a\u00e9\U0001f600"),
    (r"\s*[\r\n]+", True, "  \n \t\r\n x"),
]
ERRORS = [r"(?<=a)b", r"(a)\1", r"(?:a*)*", r"\bx", r"[[:alpha:]]", r"\p{Han}", r"(a", r"a)", r"[a", r"*a",
          r"a{3,2}", r"\q"]


def esc(s):
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif 0x20 <= o < 0x7F:
            out.append(ch)
        else:
            out.append("\\u{%X}" % o)
    return "".join(out)


def unbrace(pattern):
    """\\x{HEX} (Oniguruma and Liyab) as the character itself, for Python's regex."""
    return regex.sub(r"\\x\{([0-9A-Fa-f]+)\}", lambda m: chr(int(m.group(1), 16)), pattern)


def py_split(pattern, keep_gaps, text):
    spans, gap = [], 0
    for m in regex.finditer(unbrace(pattern), text):
        if m.end() == m.start():
            continue
        if m.start() > gap and keep_gaps:
            spans.append((gap, m.start()))
        spans.append((m.start(), m.end()))
        gap = m.end()
    if gap < len(text) and keep_gaps:
        spans.append((gap, len(text)))
    return spans


def hf_split(pattern, keep_gaps, text):
    pattern_hf = unbrace(pattern)
    split = (pre_tokenizers.Split(Regex(pattern_hf), "isolated") if keep_gaps
             else pre_tokenizers.Split(Regex(pattern_hf), "removed", invert=True))
    return [span for _, span in split.pre_tokenize_str(text)]


def hf_pre(name, text):
    steps, _ = PRE_TOKENIZERS[name]
    seq = []
    for pattern, keep_gaps in steps:
        p = unbrace(pattern)
        seq.append(pre_tokenizers.Split(Regex(p), "isolated") if keep_gaps
                   else pre_tokenizers.Split(Regex(p), "removed", invert=True))
    return [span for _, span in pre_tokenizers.Sequence(seq).pre_tokenize_str(text)]


def spans_text(spans):
    """Pieces as their end offsets, or start-end when a piece does not start where the one before ended."""
    out, prev = [], 0
    for a, b in spans:
        out.append(str(b) if a == prev else f"{a}-{b}")
        prev = b
    return " ".join(out)


def nfkc_records(edge, conform):
    """Source Unicode 9 expectations and the Digits -> ByteLevel pipeline."""
    rows = [f"# NFKC expectations (tokenizers Unicode 9); gpt2-nfkc-digits splits"]
    def escaped(text):
        return esc(text).replace(" ", r"\u{20}")
    changed = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        text = chr(cp)
        normalized = normalizers.NFKC().normalize_str(text)
        if text != normalized:
            rows.append(f"K\t{escaped(text)}\t{escaped(normalized)}")
            changed.append(text)
    mixed = ["", "plain ASCII 123", "ＡＢＣ１２３，", "ÅK ﬃ Ⅳ ½ ²", "ㄱㅏᆨ", "ｶﾞ", "\uFDFA",
             "A\u030A\u0323", "\u00A8\u0301", "\u212B\u0323"]
    rng = random.Random(2048)
    pool = changed + ["A", "e", "\u030A", "\u0301", "\u0323", "\u1100", "\u1161", "\u11A8"]
    for _ in range(256):
        mixed.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 8))))
    for text in mixed:
        if text:
            rows.append(f"K\t{escaped(text)}\t{escaped(normalizers.NFKC().normalize_str(text))}")
    source = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                     pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    normalization = normalizers.NFKC()
    for i, text in enumerate(edge.splitlines(keepends=True) + [conform[:600]] + mixed[:10]):
        if not text:
            continue
        normalized = normalizers.NFKC().normalize_str(text)
        assert normalization.normalize_str(text) == normalized
        rows.append(f"T\tnfkc_{i}\t{escaped(normalized)}")
        spans = [span for _, span in source.pre_tokenize_str(normalized)]
        rows.append(f"S\tgpt2-nfkc-digits\tnfkc_{i}\t{spans_text(spans)}")
    return rows


def main():
    edge = open(os.path.join(ROOT, "tests/data/tokenizer_edge.txt"), encoding="utf-8", newline="").read()
    conform = open(os.path.join(ROOT, "tests/data/conform_text.txt"), encoding="utf-8", newline="").read()
    texts = edge.splitlines(keepends=True) + [conform[:600]]
    out = [f"# Generated by tools/gen_text_tests.py (Unicode {unicodedata.unidata_version}, regex "
           f"{regex.__version__}). Do not edit."]
    ids = {}

    def text_id(t):
        if t not in ids:
            ids[t] = len(ids)
            out.append(f"T\t{ids[t]}\t{esc(t)}")
        return ids[t]

    pattern_ids = {}

    def pattern_id(p):
        if p not in pattern_ids:
            pattern_ids[p] = len(pattern_ids)
            out.append(f"P\t{pattern_ids[p]}\t{esc(p)}")
        return pattern_ids[p]

    disagreements = 0
    patterns = sorted({p for steps, _ in PRE_TOKENIZERS.values() for p, _ in steps})
    for pattern in patterns:
        keep = all(k for steps, _ in PRE_TOKENIZERS.values() for p, k in steps if p == pattern)
        for t in texts:
            py, hf = py_split(pattern, keep, t), hf_split(pattern, keep, t)
            disagreements += py != hf
            out.append(f"R\t{pattern_id(pattern)}\t{int(keep)}\t{text_id(t)}\t{spans_text(hf)}")
    for pattern, keep, t in FEATURES:
        py, hf = py_split(pattern, keep, t), hf_split(pattern, keep, t)
        if py != hf:
            disagreements += 1
            print(f"note: {pattern!r} on {t!r}: python {py} tokenizers {hf}", file=sys.stderr)
        out.append(f"R\t{pattern_id(pattern)}\t{int(keep)}\t{text_id(t)}\t{spans_text(hf)}")
    for pattern in ERRORS:
        out.append(f"E\t{esc(pattern)}")
    for name, (_, nfc) in PRE_TOKENIZERS.items():
        for t in texts:
            norm = unicodedata.normalize("NFC", t) if nfc else t
            out.append(f"S\t{name}\t{text_id(norm)}\t{spans_text(hf_pre(name, norm))}")

    # NFC: the edge corpus, then random mixes of starters, marks, Hangul jamo and singletons.
    rng = random.Random(1234)
    pool = ([chr(c) for c in range(0x41, 0x5B)] + ["\u00e9", "\u00c5", "\u212b", "\u2126", "\u212a", "\u1e0b",
            "\u0344", "\u0f73", "\u0958", "\u2adc", "\u1100", "\u1161", "\u11a8", "\uac00", "\uac01", "\u0301",
            "\u0300", "\u0302", "\u0323", "\u0327", "\u0308", "\u0345", "\u05b8", "\u05c1", "\u0b47", "\u0b3e",
            "\u0cc6", "\u0cd5", "\u1025", "\u102e", "\u0e40", "\u3099", "\u304b", "\U0001d15e", "\U0001d165",
            "\u0340", "\u00a8", " "])
    cases = [line for line in edge.splitlines()]
    for _ in range(300):
        cases.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 8))))
    for s in cases:
        out.append(f"N\t{esc(s)}\t{esc(unicodedata.normalize('NFC', s))}")
    out.extend(nfkc_records(edge, conform))
    # The SSM tokenizers' original Rust pipelines, rather than a second copy
    # of the regex table. Keep these additive so existing pattern/text ids
    # and their pinned expectations remain stable.
    ssm_pre = {
        "olmo": pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        "falcon": pre_tokenizers.Sequence([
            pre_tokenizers.Punctuation(behavior="contiguous"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
            pre_tokenizers.Digits(individual_digits=False),
            pre_tokenizers.Split(Regex("[0-9][0-9][0-9]"), behavior="isolated"),
        ]),
    }
    for name, pre in ssm_pre.items():
        for t in texts:
            # The Rust offsets refer to whole original characters even when
            # a later step cuts their encoded bytes. Decode the actual
            # pieces to bytes to establish their exact boundaries instead.
            alphabet = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
            chars = alphabet + [256 + i for i in range(256 - len(alphabet))]
            byte_values = alphabet + [b for b in range(256) if b not in alphabet]
            decoder = dict(zip(map(chr, chars), byte_values))
            spans, pos = [], 0
            for piece, _ in pre.pre_tokenize_str(t):
                raw = bytes(decoder[c] for c in piece)
                assert t.encode()[pos:pos + len(raw)] == raw
                spans.append((pos, pos + len(raw)))
                pos += len(raw)
            assert pos == len(t.encode())
            out.append(f"B\t{name}\t{text_id(t)}\t{spans_text(spans)}")
    print("\n".join(out))
    print(f"{disagreements} split(s) where Python's regex and the tokenizers differ (tokenizers kept)",
          file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nfkc-only", action="store_true", help="write only the additive NFKC and source-split records")
    args = parser.parse_args()
    if args.nfkc_only:
        edge = open(os.path.join(ROOT, "tests/data/tokenizer_edge.txt"), encoding="utf-8", newline="").read()
        conform = open(os.path.join(ROOT, "tests/data/conform_text.txt"), encoding="utf-8", newline="").read()
        print("\n".join(nfkc_records(edge, conform)))
    else:
        main()
