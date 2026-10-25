#!/usr/bin/env python3
"""Describes a model for Liyab from its Hugging Face implementation and a GGUF of it, and verifies the description.

    tools/describe_model.py --model Qwen/Qwen2.5-1.5B-Instruct --gguf Qwen2.5-1.5B-Instruct-f16.gguf --gate
    tools/describe_model.py --model unsloth/gemma-2-2b-it --gguf hf://bartowski/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q8_0.gguf

Liyab describes an architecture by one row of facts its tensors cannot tell (src/core/architecture.cpp, ArchSpec)
and plans each block from the tensors present. This tool establishes those facts for a model and either prints the
registry row it needs or the precise list of parts Liyab does not implement yet. It never guesses: a fact it cannot
establish is reported as unknown, and no row is proposed. Where this GGUF disagrees with the reference (other
weights, a window or BOS rule it does not record), that is reported as a problem of the file, apart from the row.

Where the facts come from:

* The reference implementation itself (tools/reference_trace.py): a reduced copy of the Hugging Face model with
  random weights runs once while every module's inputs and outputs are recorded, and each fact is recomputed from
  them: the order of norms, mixer and FFN and how the residual adds combine them, each norm's formula, which key
  dimensions RoPE pairs and with which frequencies, where Q/K norms sit, the attention's scale, window and soft-cap,
  the FFN's activation, the MoE routing, how embeddings enter and logits leave.
* The GGUF's metadata and tensor names (and shapes): hyperparameters, the per-block plan (mixer, fused or separate
  projections, gated attention, dense or MoE FFN, shared expert), the RoPE encoding the file carries (base, scaling,
  frequency factors, YaRN), the window, the tokenizer and its BOS and space-prefix rules. Every tensor must be one
  the engine binds.
* A few weights of both checkpoints, read with HTTP range requests when remote (a few MB): two heads of one block's
  key projection, which tell whether the converter permuted its rows (so how RoPE pairs dimensions in the file),
  and one block's norm weights, which tell which GGUF tensor holds which norm and whether a norm the reference
  applies as x * (1 + w) is stored as 1 + w.
* The engine's own sources: the ArchSpec fields and enums (architecture.h), the registry rows and the metadata keys
  and RoPE scaling types describe_model() reads (architecture.cpp: what the engine takes from the file, such as an
  attention scale, soft caps or YaRN, is supported exactly when it reads it), the tensors the forward pass binds
  (transformer.cpp), the pre-tokenizer table and tokenizer kinds.

With --gate the row is then verified: the tool copies the source tree to a scratch directory, adds or replaces the
row there (the repository is never modified), builds liyab-kl and liyab-tok, and runs the conformance gate:
liyab-tok's ids against the model's own tokenizer on three texts (tools/tokenizer_reference.py), then liyab-kl
--ref-tokens --kv f16 against transformers in float32 (tools/conform_reference.py, eager attention when the model
soft-caps its scores and that tool takes --attn-implementation) with --max-kl and --min-top1, which must pass and
must find the file's own encoding of the text, BOS included, equal to the reference's. The patch is reverted
afterwards, so the scratch tree stays a pristine copy and later runs rebuild only what changed. The row is accepted
only if every check passes. The gate needs the GGUF as a local file (a BF16/F16 one: lower bit widths add their own
error) and downloads the Hugging Face weights.

Needs torch, transformers, huggingface_hub, numpy and gguf (for quantized tensors). Run with the toolchain on PATH
(cmake, ninja) for --gate. --self-test checks the registry parser and printer against architecture.cpp (no network).
Exit status: 0 when a row is proposed (and accepted with --gate), 1 when the model needs parts Liyab lacks, a fact
is unknown or the gate fails, 2 on usage errors or a file that cannot be read as a GGUF.
"""

import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # reference_trace.py, also under python -I

# ----------------------------------------------------------------------------------------------------------------
# Byte sources: local files, or files of a Hugging Face repository read with HTTP range requests
# ----------------------------------------------------------------------------------------------------------------


class ByteSource:
    """Random access to a local file or to hf://<repo>/<path> (optionally hf://<repo>@<revision>/<path>)."""

    def __init__(self, spec):
        self.spec = spec
        if spec.startswith("hf://"):
            from huggingface_hub import HfFileSystem
            # Small blocks: only a header and a few tensor rows are read.
            self.file = HfFileSystem().open(spec[len("hf://"):], "rb", block_size=256 << 10)
            self.size = int(self.file.size)
        else:
            self.file = open(spec, "rb")
            self.size = os.fstat(self.file.fileno()).st_size

    def read_at(self, offset, size):
        self.file.seek(offset)
        data = self.file.read(size)
        if len(data) != size:
            raise IOError(f"{self.spec}: short read at {offset}")
        return data

    def sibling(self, name):
        """The file `name` in the same directory or repository folder."""
        if self.spec.startswith("hf://"):
            return ByteSource(self.spec.rsplit("/", 1)[0] + "/" + name)
        return ByteSource(os.path.join(os.path.dirname(self.spec), name))


# ----------------------------------------------------------------------------------------------------------------
# GGUF: metadata, tensor infos, and the values of a few tensors
# ----------------------------------------------------------------------------------------------------------------

class FormatError(ValueError):
    """A file that is not a GGUF this tool can read."""


GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
GGML_F32, GGML_F16, GGML_BF16 = 0, 1, 30


class _Cursor:
    """Sequential reads over a ByteSource, fetched in 1 MiB chunks (a remote header is read in a few requests)."""

    def __init__(self, src, offset=0):
        self.src, self.pos, self.buf, self.base = src, offset, b"", offset

    def read(self, n):
        while self.pos + n > self.base + len(self.buf):
            start = self.base + len(self.buf)
            want = min(max(1 << 20, self.pos + n - start), self.src.size - start)
            if want <= 0:
                raise FormatError(f"{self.src.spec}: ends inside its GGUF header")
            self.buf += self.src.read_at(start, want)
        out = self.buf[self.pos - self.base:self.pos - self.base + n]
        self.pos += n
        return out

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))

    def string(self):
        (n,) = self.unpack("Q")
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vtype):
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype, count = self.unpack("IQ")
            if etype == 8:
                return [self.string() for _ in range(count)]
            if etype in GGUF_SCALARS:
                size = struct.calcsize("<" + GGUF_SCALARS[etype])
                return list(struct.unpack(f"<{count}{GGUF_SCALARS[etype]}", self.read(size * count)))
            return [self.value(etype) for _ in range(count)]
        if vtype in GGUF_SCALARS:
            return self.unpack(GGUF_SCALARS[vtype])[0]
        raise FormatError(f"unknown GGUF value type {vtype}")


class Gguf:
    """A GGUF's metadata and tensor infos (all parts of a split model), and access to tensor values."""

    def __init__(self, spec):
        self.spec = spec
        first = ByteSource(spec)
        self.metadata, self.tensors = {}, {}
        self._read_part(first)
        count = int(self.metadata.get("split.count", 1) or 1)
        if count > 1:
            m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(spec.split("@")[-1]))
            if not m:
                raise FormatError(f"{spec}: split.count {count} but the name has no -0000N-of-0000M part")
            for i in range(2, count + 1):
                self._read_part(first.sibling(f"{m.group(1)}-{i:05d}-of-{count:05d}.gguf"))

    def _read_part(self, src):
        cur = _Cursor(src)
        if cur.read(4) != b"GGUF":
            raise FormatError(f"{src.spec} is not a GGUF file")
        version, n_tensors, n_kv = cur.unpack("IQQ")
        if version not in (2, 3):
            raise FormatError(f"{src.spec}: GGUF version {version} is not supported")
        for _ in range(n_kv):
            key = cur.string()
            (vtype,) = cur.unpack("I")
            value = cur.value(vtype)
            if not (key.startswith("split.") and key in self.metadata):
                self.metadata.setdefault(key, value)
        infos = []
        for _ in range(n_tensors):
            name = cur.string()
            (n_dims,) = cur.unpack("I")
            dims = cur.unpack("Q" * n_dims)
            ttype, offset = cur.unpack("IQ")
            infos.append((name, tuple(dims), ttype, offset))
        align = int(self.metadata.get("general.alignment", 32))
        data_start = (cur.pos + align - 1) // align * align
        for name, dims, ttype, offset in infos:
            self.tensors[name] = {"dims": dims, "type": ttype, "offset": data_start + offset, "source": src}

    @property
    def arch(self):
        return self.metadata.get("general.architecture")

    def key(self, suffix, default=None):
        return self.metadata.get(f"{self.arch}.{suffix}", default)

    def rows(self, name):
        dims = self.tensors[name]["dims"]
        return int(math.prod(dims[1:])) if len(dims) > 1 else 1

    def cols(self, name):
        return int(self.tensors[name]["dims"][0])

    def values(self, name, first_row=0, rows=None):
        """The tensor (or `rows` of its rows from `first_row`) as a float32 numpy array [rows, cols]."""
        import numpy as np
        t = self.tensors[name]
        cols = int(t["dims"][0])
        n_rows = self.rows(name) - first_row if rows is None else rows
        ttype = t["type"]
        if ttype in (GGML_F32, GGML_F16, GGML_BF16):
            size = {GGML_F32: 4, GGML_F16: 2, GGML_BF16: 2}[ttype]
            raw = t["source"].read_at(t["offset"] + first_row * cols * size, n_rows * cols * size)
            if ttype == GGML_F32:
                arr = np.frombuffer(raw, dtype="<f4")
            elif ttype == GGML_F16:
                arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            else:
                arr = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            return arr.reshape(n_rows, cols).astype(np.float32)
        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
        from gguf.quants import dequantize
        qtype = GGMLQuantizationType(ttype)
        block, type_size = GGML_QUANT_SIZES[qtype]
        row_bytes = cols // block * type_size
        raw = t["source"].read_at(t["offset"] + first_row * row_bytes, n_rows * row_bytes)
        data = np.frombuffer(raw, dtype=np.uint8).reshape(n_rows, row_bytes)
        return dequantize(data, qtype).reshape(n_rows, cols).astype(np.float32)


def block_index(name):
    m = re.match(r"blk\.(\d+)\.(.+)$", name)
    return (int(m.group(1)), m.group(2)) if m else (None, name)


# ----------------------------------------------------------------------------------------------------------------
# Hugging Face checkpoint: a few tensors, by name suffix
# ----------------------------------------------------------------------------------------------------------------


class Checkpoint:
    """Safetensors tensors of a Hugging Face model (local directory or repository), read on demand."""

    DTYPES = {"F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": ("<u2", 2)}

    def __init__(self, model, revision=None):
        self.model, self.revision = model, revision
        self.files = {}  # shard name -> (source, header dict, data start)
        self.index = {}  # tensor name -> shard name
        index = self._json("model.safetensors.index.json")
        if index is not None:
            self.index = dict(index["weight_map"])
        else:
            for name in self._header("model.safetensors")[0]:
                if name != "__metadata__":
                    self.index[name] = "model.safetensors"

    def _path(self, name):
        if os.path.isdir(self.model):
            path = os.path.join(self.model, name)
            return path if os.path.exists(path) else None
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(self.model, name, revision=self.revision)
        if isinstance(cached, str):
            return cached
        rev = f"@{self.revision}" if self.revision else ""
        return f"hf://{self.model}{rev}/{name}"

    def _json(self, name):
        path = self._path(name)
        if path is None:
            return None
        try:
            src = ByteSource(path)
            src.file.seek(0)
            return json.loads(src.file.read())
        except (FileNotFoundError, OSError):
            return None

    def _header(self, shard):
        if shard not in self.files:
            src = ByteSource(self._path(shard))
            (size,) = struct.unpack("<Q", src.read_at(0, 8))
            header = json.loads(src.read_at(8, size))
            self.files[shard] = (src, header, 8 + size)
        _, header, _ = self.files[shard]
        return header, shard

    def find(self, suffix):
        """The tensor name ending in `suffix` (the shortest when a vision or MTP copy also matches), or None."""
        hits = [n for n in self.index if n.endswith(suffix) and (n == suffix or n[-len(suffix) - 1] == ".")]
        hits = [n for n in hits if not re.search(r"(^|\.)(mtp|visual|vision_tower|vision_model)\.", n)] or hits
        return min(hits, key=len) if hits else None

    def tensor(self, name, first_row=0, rows=None):
        """The tensor, or `rows` of its rows from `first_row` (rows of the leading dimension), as float32."""
        import numpy as np
        shard = self.index[name]
        header, _ = self._header(shard)
        src, _, start = self.files[shard]
        info = header[name]
        dtype, size = self.DTYPES[info["dtype"]]
        begin, end = info["data_offsets"]
        shape = list(info["shape"])
        row_bytes = (end - begin) // max(shape[0], 1)
        rows = shape[0] - first_row if rows is None else rows
        raw = src.read_at(start + begin + first_row * row_bytes, rows * row_bytes)
        arr = np.frombuffer(raw, dtype=dtype)
        if info["dtype"] == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32).reshape([rows] + shape[1:])


# ----------------------------------------------------------------------------------------------------------------
# The engine's sources: ArchSpec, registry, bound tensors, tokenizers
# ----------------------------------------------------------------------------------------------------------------


def _strip_comments(text):
    return re.sub(r"//[^\n]*", "", text)


def _split_top(text, sep=","):
    """Splits on `sep` outside braces, parentheses and string literals."""
    parts, depth, cur, quote = [], 0, "", False
    for i, ch in enumerate(text):
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            quote = not quote
        elif not quote and ch in "{(":
            depth += 1
        elif not quote and ch in "})":
            depth -= 1
        if ch == sep and depth == 0 and not quote:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


class Engine:
    """What the engine's sources say: ArchSpec fields with defaults, enum values, registry rows, the tensors the
    forward pass binds, the pre-tokenizer names and the tokenizer kinds."""

    def __init__(self, root):
        self.root = root
        read = lambda p: open(os.path.join(root, p), encoding="utf-8").read()
        header = read("src/core/architecture.h")
        self.enums = {}
        for m in re.finditer(r"enum class (\w+)\s*:\s*\w+\s*\{(.*?)\};", header, re.S):
            self.enums[m.group(1)] = [v.strip() for v in _strip_comments(m.group(2)).split(",") if v.strip()]
        body = re.search(r"struct ArchSpec \{(.*?)\n\};", header, re.S)
        if body is None:
            raise ValueError("architecture.h: struct ArchSpec not found")
        self.fields = []  # (name, type, default or None)
        for decl in _strip_comments(body.group(1)).split(";"):
            decl = " ".join(decl.split())
            if not decl:
                continue
            m = re.match(r"([\w:<>]+) (\w+)(?: = (.+))?$", decl)
            if m is None:
                raise ValueError(f"architecture.h: cannot read the ArchSpec field '{decl}'")
            ftype, name, default = m.groups()
            self.fields.append((name, ftype, None if default is None else self.parse_value(ftype, default)))
        cpp = read("src/core/architecture.cpp")
        # What describe_model() reads from a file: its metadata keys (key("...") and require_int("...")) and
        # the RoPE scaling types it accepts. Features the engine takes from the file (an attention scale, soft
        # caps, YaRN) are supported exactly when it reads them.
        self.file_keys = set(re.findall(r'(?:key|require_int)\(\s*"([^"]+)"', cpp))
        self.rope_scalings = set(re.findall(r'\*scaling\s*[!=]=\s*"([\w-]+)"', cpp))
        m = re.search(r"kArchitectures\[\]\s*=\s*\{(.*?)\n\};", cpp, re.S)
        if m is None:
            raise ValueError("architecture.cpp: kArchitectures not found")
        self.rows = {}  # name -> (values dict, source text)
        for entry in self._entries(m.group(1)):
            values = self.parse_row(entry)
            self.rows[values["name"]] = (values, entry)
        transformer = read("src/core/transformer.cpp")
        self.block_tensors = set(re.findall(r'block_tensor\(\s*\w+\s*,\s*"([^"]+)"\s*\)', transformer))
        self.block_tensors |= set(re.findall(r'experts\(\s*"([^"]+)"', transformer))
        literals = set(re.findall(r'"([a-z][a-z0-9_]*\.(?:weight|bias))"', transformer))
        self.global_tensors = {t for t in literals if t not in self.block_tensors}
        pre = read("src/core/pre_tokenizer.cpp")
        table = re.search(r"table\(\)\s*\{(.*?)\n\}", pre, re.S)
        self.pre_tokenizers = []
        for names in re.findall(r'\{\{("[^"]+"(?:\s*,\s*"[^"]+")*)\}', table.group(1) if table else ""):
            self.pre_tokenizers += re.findall(r'"([^"]+)"', names)
        self.tokenizer_kinds = re.findall(r'\*kind == "(\w+)"', read("src/core/tokenizer.cpp"))

    @staticmethod
    def _entries(text):
        out, depth, start, quote = [], 0, None, False
        text = _strip_comments(text)
        for i, ch in enumerate(text):
            if ch == '"' and text[i - 1] != "\\":
                quote = not quote
            if quote:
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start:i + 1])
        return out

    def parse_value(self, ftype, text):
        text = text.strip()
        if ftype == "std::string_view":
            return "" if text == "{}" else json.loads(text)
        if ftype == "bool":
            return {"true": True, "false": False}[text]
        if ftype in ("int32_t", "int", "uint32_t", "int64_t"):
            return int(text)
        if ftype in ("float", "double"):
            return float(text.rstrip("f"))
        if ftype in self.enums:
            name = text.split("::")[-1]
            if name not in self.enums[ftype]:
                raise ValueError(f"{text} is not a {ftype} value")
            return name
        raise ValueError(f"ArchSpec field type {ftype} is not known to this tool")

    def field_type(self, name):
        return next(t for n, t, _ in self.fields if n == name)

    def parse_row(self, entry):
        items = _split_top(entry.strip()[1:-1])
        values = {n: d for n, _, d in self.fields}
        for i, item in enumerate(items):
            m = re.match(r"\.(\w+)\s*=\s*(.+)$", item, re.S)
            name, text = (m.group(1), m.group(2)) if m else (self.fields[i][0], item)
            values[name] = self.parse_value(self.field_type(name), text)
        return values

    def format_value(self, name, value):
        ftype = self.field_type(name)
        if ftype == "std::string_view":
            return json.dumps(value)
        if ftype == "bool":
            return "true" if value else "false"
        if ftype in ("float", "double"):
            text = repr(float(value))
            return (text if ("." in text or "e" in text) else text + ".0") + "f"
        if ftype in self.enums:
            return f"{ftype}::{value}"
        return str(value)

    def format_row(self, values):
        """The row as architecture.cpp writes rows: positional while the non-default fields after the required
        ones form a prefix of the optional fields, designated otherwise (one field per line past 120 columns)."""
        required = [n for n, _, d in self.fields if d is None]
        optional = [n for n, _, d in self.fields if d is not None]
        default = {n: d for n, _, d in self.fields}
        changed = [n for n in optional if values[n] != default[n]]
        prefix = optional[:len(changed)]
        if changed == prefix:
            return "    {" + ", ".join(self.format_value(n, values[n]) for n in required + prefix) + "},"
        entries = [f".{n} = {self.format_value(n, values[n])}" for n in required + changed]
        line = "    {" + ", ".join(entries) + "},"
        if len(line) <= 120:
            return line
        return "    {" + ",\n     ".join(entries) + "},"


# ----------------------------------------------------------------------------------------------------------------
# The description
# ----------------------------------------------------------------------------------------------------------------

# Activations the reference trace names, and the FfnActivation value that runs each (when the enum has it).
ACTIVATION_ENUM = {"silu": "SiLU", "gelu_tanh": "GeluTanh", "gelu": "Gelu", "relu": "Relu", "relu2": "ReluSquared",
                   "gelu_quick": "GeluQuick", "xielu": "XIelu"}
QK_ORDER_ENUM = {"before_rope": "BeforeRope", "after_rope": "AfterRope"}
LAYOUT_ENUM = {"halves": "Halves", "adjacent": "Adjacent"}
# The DeltaNet tensors the engine binds (transformer.cpp, bind_weights), present together in a recurrent block.
DELTANET_TENSORS = {"attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight",
                    "ssm_conv1d.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight"}
# The fields this tool derives. A field architecture.h gains needs a rule here before rows are proposed again.
DERIVED_FIELDS = ("name", "rope", "ffn_norm", "moe_norm_weights", "qk_norm", "ffn_activation", "post_mixer_norm",
                  "post_ffn_norm", "scale_embeddings", "swa_period", "rope_base_swa", "norm", "parallel_residual",
                  "rotary", "alibi_bias", "sliding_window_default", "spm_prefix_after_control", "spm_add_bos_default",
                  "spm_add_space_prefix_default", "spm_charsmap", "spm_user_defined")


def _close(a, b, rel=1e-5):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-30)


def _ranges(indices):
    """[0, 1, 2, 4] -> "0-2, 4"."""
    out, start, prev = [], None, None
    for i in sorted(indices):
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = i
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(out)


class Description:
    def __init__(self):
        self.fields = {}  # field -> (value, evidence)
        self.parts = []  # per-file parts the model uses (informational)
        self.missing = []  # parts Liyab does not implement
        self.unknown = []  # facts that could not be established
        self.file_issues = []  # the GGUF disagrees with the reference
        self.notes = []
        self.row = None
        self.registry = None  # comparison with the existing row
        self.softcap = False  # the reference soft-caps attention scores (its reference run needs eager attention)

    def set(self, field, value, evidence):
        self.fields[field] = (value, evidence)

    @property
    def ok(self):
        """Every fact established and every part implemented. Disagreements of this GGUF with the reference
        (file_issues) concern the file, not the architecture's row: they are reported, and the gate fails on
        them where they matter."""
        return not (self.missing or self.unknown)

    def merge_blocks(self):
        """"block 0: X", "block 1: X" -> "X (blocks 0-1)": a fact of every traced block is said once."""
        self.parts = list(dict.fromkeys(self.parts))
        self.notes = list(dict.fromkeys(self.notes))
        for name in ("missing", "unknown", "file_issues"):
            merged, order = {}, []
            for item in getattr(self, name):
                m = re.match(r"block (\d+): (.*)$", item, re.S)
                text, block = (m.group(2), int(m.group(1))) if m else (item, None)
                if text not in merged:
                    merged[text] = []
                    order.append(text)
                if block is not None:
                    merged[text].append(block)
            setattr(self, name, [t + (f" ({'block' if len(merged[t]) == 1 else 'blocks'} {_ranges(set(merged[t]))})"
                                      if merged[t] else "") for t in order])

    def to_json(self):
        return {"fields": {k: {"value": v, "evidence": e} for k, (v, e) in self.fields.items()},
                "parts": self.parts, "missing": self.missing, "unknown": self.unknown,
                "file_issues": self.file_issues, "notes": self.notes, "row": self.row, "registry": self.registry}


def _rel_err(a, b):
    import numpy as np
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def rope_permutation(gguf_rows, hf_rows, heads, head_dim):
    """Which row order the converter wrote a projection's heads in, from the weights: "identity", "interleave"
    (row 2i + s of a head is the reference's row s * d/2 + i: halves become adjacent pairs) or "deinterleave"
    (the inverse). Returns (name, relative error, runner-up error) or None when no order matches."""
    import numpy as np
    d = head_dim
    idx = np.arange(heads * d).reshape(heads, d)
    inter = np.empty_like(idx)
    inter[:, 0::2] = idx[:, : d // 2]
    inter[:, 1::2] = idx[:, d // 2:]
    deinter = np.empty_like(idx)
    deinter[:, : d // 2] = idx[:, 0::2]
    deinter[:, d // 2:] = idx[:, 1::2]
    errs = sorted((_rel_err(gguf_rows, hf_rows[p.reshape(-1)]), name)
                  for name, p in (("identity", idx), ("interleave", inter), ("deinterleave", deinter)))
    (best, name), (second, _) = errs[0], errs[1]
    # Float files match to rounding; quantized ones to their error, far below a wrong order's.
    if best < 0.3 and second > 3 * best and second > 0.5:
        return name, best, second
    return None


def pairing_in_file(ref_layout, permutation, rope_dims, head_dim):
    """The engine's RopeLayout for a file whose rows were reordered by `permutation` from a reference that pairs
    dimensions as `ref_layout`: where each reference pair (a, b) lands in the file."""
    half = rope_dims // 2
    ref_pairs = [(i, i + half) if ref_layout == "halves" else (2 * i, 2 * i + 1) for i in range(half)]
    pos = list(range(head_dim))  # file row r holds reference row src[r]
    if permutation == "interleave":
        src = [0] * head_dim
        for i in range(head_dim // 2):
            src[2 * i], src[2 * i + 1] = i, i + head_dim // 2
    elif permutation == "deinterleave":
        src = [0] * head_dim
        for i in range(head_dim // 2):
            src[i], src[i + head_dim // 2] = 2 * i, 2 * i + 1
    else:
        src = pos
    where = {s: r for r, s in enumerate(src)}
    file_pairs = sorted(tuple(sorted((where[a], where[b]))) for a, b in ref_pairs)
    if file_pairs == sorted((i, i + half) for i in range(half)):
        return "halves"
    if file_pairs == sorted((2 * i, 2 * i + 1) for i in range(half)):
        return "adjacent"
    return None


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def engine_inv_freq(g, rope_dim, n_ctx_train, long_context=False):
    """RoPE frequencies and magnitude the engine computes from the file (Transformer::load and describe_model()):
    base^(-2i/r) divided by the file's frequency factors, blended by YaRN when the file asks for it, over the
    linear position scale. Returns (inv_freq / scale list, magnitude, q.k factor, how)."""
    base = float(g.key("rope.freq_base", 10000.0))
    scaling = g.key("rope.scaling.type")
    scale = 1.0
    magnitude = float(g.key("rope.scaling.attn_factor", 1.0))
    qk = 1.0  # YaRN's factor on q.k when the file has a yarn_log_multiplier (squared into the attention scale)
    how = [f"base {base:g}"]
    if scaling == "linear":
        scale = float(g.key("rope.scaling.factor", 1.0))
        how.append(f"linear scaling {scale:g}")
    factors = [1.0] * (rope_dim // 2)
    has_long = "rope_factors_long.weight" in g.tensors
    name = None
    if has_long:
        name = "rope_factors_long.weight" if long_context else "rope_factors_short.weight"
    elif "rope_freqs.weight" in g.tensors:
        name = "rope_freqs.weight"
    if name is not None:
        vals = g.values(name).reshape(-1)
        if len(vals) == rope_dim // 2:
            factors = [float(v) for v in vals]
            how.append(f"factors {name.split('.')[0]}")
    yarn = None
    if scaling == "yarn":
        factor = float(g.key("rope.scaling.factor", 0.0))
        ctx = float(g.key("rope.scaling.original_context_length", n_ctx_train))
        fast = float(g.key("rope.scaling.yarn_beta_fast", 32.0))
        slow = float(g.key("rope.scaling.yarn_beta_slow", 1.0))
        ext = float(g.key("rope.scaling.yarn_ext_factor", 1.0))
        truncate = bool(g.key("rope.scaling.yarn_truncate", True))
        # The correction range: the (fractional) pairs that turn beta_fast and beta_slow times over the
        # original context.
        turning = lambda turns: rope_dim * math.log(ctx / (turns * 2.0 * math.pi)) / (2.0 * math.log(base))
        low, high = turning(fast), turning(slow)
        if truncate:
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0.0), min(high, rope_dim - 1.0)
        if high == low:
            high += 0.001
        yarn = (factor, low, high, ext)
        ln = math.log(factor) if factor > 1.0 else 0.0
        log_mul = g.key("rope.scaling.yarn_log_multiplier")
        attn = g.key("rope.scaling.yarn_attn_factor")
        magnitude *= float(attn) if attn is not None else (1.0 if log_mul is not None else 0.1 * ln + 1.0)
        if log_mul is not None:
            qk = 1.0 + float(log_mul) * ln
        how.append(f"YaRN factor {factor:g} over {ctx:g} positions")
    inv = []
    for i in range(rope_dim // 2):
        original = base ** (-2.0 * i / rope_dim) / factors[i]
        if yarn is not None:
            factor, low, high, ext = yarn
            ramp = min(max((i - low) / (high - low), 0.0), 1.0)
            kept = (1.0 - ramp) * ext
            original = original / factor * (1.0 - kept) + original * kept
        inv.append(_f32(original) / scale)
    return inv, _f32(magnitude), qk, ", ".join(how)


def describe(model, gguf_spec, engine, revision=None, trust_remote_code=False):
    import numpy as np
    import reference_trace

    d = Description()
    g = Gguf(gguf_spec)
    arch = g.arch
    if not arch:
        d.unknown.append("the GGUF has no general.architecture")
        return d, None, g
    try:
        facts = reference_trace.describe_reference(model, revision, trust_remote_code)
    except Exception as e:  # noqa: BLE001 - whatever stops the reference stops the description: say what
        d.unknown.append(f"the reference implementation could not be traced: {type(e).__name__}: {e}")
        return d, None, g
    cfg = facts["config"]
    d.set("name", arch, "general.architecture of the GGUF")
    d.notes.append(f"reference: {facts['model_class']} ({facts.get('model_type')}), {facts['traced_blocks']} of "
                   f"{facts['blocks']} blocks traced")
    unknown_fields = [n for n, _, _ in engine.fields if n not in DERIVED_FIELDS]
    for n in unknown_fields:
        d.unknown.append(f"ArchSpec has a field this tool has no rule for: {n} (add one to "
                         f"tools/describe_model.py before proposing rows)")
    d.unknown += [f"reference: {u}" for u in facts["unknown"]]

    # --- hyperparameters from the file, as describe_model() reads them ---
    n_layers = int(g.key("block_count", 0)) - int(g.key("nextn_predict_layers", 0) or 0)
    n_embd = int(g.key("embedding_length", 0))
    n_head = int(g.key("attention.head_count", 0) or 0)
    n_head_kv = int(g.key("attention.head_count_kv", n_head) or n_head)
    head_dim = int(g.key("attention.key_length", n_embd // max(n_head, 1)))
    rope_dim = int(g.key("rope.dimension_count", head_dim))
    n_ff = int(g.key("feed_forward_length", 0) or 0)
    eps = float(g.key("attention.layer_norm_epsilon", g.key("attention.layer_norm_rms_epsilon", 1e-5)))
    if n_layers != facts["blocks"]:
        d.file_issues.append(f"the GGUF has {n_layers} blocks, the reference {facts['blocks']}")
    ref_embd = next((cfg[k] for k in ("hidden_size", "n_embd", "d_model", "model_dim") if cfg.get(k)), None)
    if n_embd != ref_embd:
        d.file_issues.append(f"embedding_length {n_embd}, the reference's residual width {ref_embd}")
    vlen = g.key("attention.value_length")
    if vlen is not None and int(vlen) != head_dim:
        d.missing.append(f"value heads of {vlen} dimensions with key heads of {head_dim} "
                         f"(attention.value_length != key_length)")

    # --- tensors: every one must be bound by the engine ---
    blocks = {}
    globals_ = set()
    for name in g.tensors:
        b, suffix = block_index(name)
        if b is None:
            globals_.add(name)
        else:
            blocks.setdefault(b, set()).add(suffix)
    skipped = sorted(b for b in blocks if b >= n_layers)
    if skipped:
        d.parts.append(f"blocks {_ranges(skipped)}: multi-token-prediction heads, skipped by the main pass")

    def suffix_rows(b, s):
        name = f"blk.{b}.{s}"
        return g.rows(name) if name in g.tensors else None

    # Per-block plan, as describe_model() makes it from the tensors present.
    q_dim = n_head * head_dim
    plans = []
    for b in range(n_layers):
        ts = blocks.get(b, set())
        p = {}
        if "ssm_conv1d.weight" in ts:
            p["mixer"] = "deltanet"
        else:
            p["mixer"] = "attention"
            p["attn_gate"] = suffix_rows(b, "attn_q.weight") == 2 * q_dim
            p["fused_qkv"] = "attn_q.weight" not in ts and "attn_qkv.weight" in ts
            p["qkv_bias"] = bool({"attn_q.bias", "attn_qkv.bias"} & ts)
            p["qk_norm"] = "attn_k_norm.weight" in ts
        if "ffn_gate_inp.weight" in ts:
            p["ffn"] = "moe"
            p["shared_expert"] = "ffn_up_shexp.weight" in ts
            p["shared_gate"] = "ffn_gate_inp_shexp.weight" in ts
            p["selection_bias"] = "exp_probs_b.bias" in ts
            p["fused_experts"] = "ffn_gate_up_exps.weight" in ts
        else:
            p["ffn"] = "dense"
            p["fused_gate_up"] = "ffn_gate.weight" not in ts and suffix_rows(b, "ffn_up.weight") == 2 * n_ff
        plans.append(p)

    # --- the reference, block by block ---
    bf = facts["block_facts"]
    att_blocks = [b for b in bf if b.get("mixer", {}).get("kind") == "attention"]
    rec_blocks = [b for b in bf if b.get("mixer", {}).get("kind") == "recurrent"]
    for b in bf:
        for u in b.get("unknown", []):
            d.unknown.append(f"block {b['index']}: {u}")
        for a in b.get("absent", []):
            if b.get("parallel") and "same normed input" in a:
                continue  # the shared-input form of a parallel block
            d.missing.append(f"block {b['index']}: {a}")
        i = b["index"]
        if i < len(plans):
            kind = b.get("mixer", {}).get("kind")
            if kind == "attention" and plans[i]["mixer"] != "attention":
                d.file_issues.append(f"block {i}: the reference attends, the GGUF has a recurrent block")
            if kind == "recurrent" and plans[i]["mixer"] != "deltanet":
                d.missing.append(f"block {i}: a token mixer that is not softmax attention "
                                 f"({', '.join(b['mixer'].get('classes', []))}) without the engine's Gated "
                                 f"DeltaNet tensors")

    # Unknown mixers: say what the file holds, e.g. latent attention.
    for b in att_blocks:
        m = b["mixer"]
        for u in m.get("unknown", []):
            d.unknown.append(f"block {b['index']}: {u}")
    mla = {"attn_kv_a_mqa.weight", "attn_kv_b.weight", "attn_k_b.weight", "attn_v_b.weight", "attn_q_a.weight"}
    if any(mla & blocks.get(i, set()) for i in range(n_layers)):
        if "attn_kv_a_mqa.weight" in engine.block_tensors:
            # The engine runs latent attention from the file's tensors and geometry, but this tool has no rule
            # comparing them with the reference's: it cannot say the model runs, so it proposes no row.
            d.unknown.append("multi-head latent attention: the engine runs it, but its facts are not compared with "
                             "the reference by this tool; check the model with liyab-kl")
        else:
            d.missing.append("multi-head latent attention: keys and values decompressed from a shared latent "
                             "(attn_kv_a_mqa, attn_kv_a_norm, attn_kv_b), with a separate rotary part")

    # --- RoPE: pairing (reference + converter), frequencies, NoPE blocks ---
    layouts = set()
    for b in att_blocks:
        m = b["mixer"]
        for side in ("key", "query"):
            info = m.get(side)
            if info is None:
                continue
            if info["rope"] is not None:
                layouts.add(info["rope"]["layout"])
    rotated = {(m.get(side) or {}).get("rope") is not None for b in att_blocks for m in [b["mixer"]]
               for side in ("key", "query") if m.get(side) is not None}
    if len(rotated) == 1:
        d.set("rotary", next(iter(rotated)), "queries and keys are rotated" if True in rotated else
              "queries and keys are passed without rotation")
        if False in rotated:
            d.set("rope", engine.enums["RopeLayout"][0], "no rotary dimensions (layout is unused)")
    elif rotated:
        d.missing.append("rotary attention in only some blocks; the architecture row applies rotary to every block")
    ref_layout = layouts.pop() if len(layouts) == 1 else None
    if len(layouts) > 0:
        d.unknown.append("the reference pairs RoPE dimensions differently in different blocks")
    ckpt = None
    try:
        ckpt = Checkpoint(model, revision)
    except Exception as e:  # noqa: BLE001 - reported as unknown
        d.notes.append(f"reference weights unavailable ({type(e).__name__}: {e})")

    def layer_suffix(path):
        """'model.layers.3.self_attn.k_proj' -> 'layers.3.self_attn.k_proj' (checkpoint names vary in prefix)."""
        m = re.search(r"(?:^|\.)([^.]+\.\d+\..+)$", path)
        return m.group(1) if m else path

    file_layout = None
    rope_tables = {t["name"]: t for t in facts["rope_tables"]}
    first_att = next((b for b in att_blocks if b["mixer"].get("key") and b["mixer"]["key"].get("rope")), None)
    if first_att is not None and ref_layout is not None:
        key = first_att["mixer"]["key"]
        i = first_att["index"]
        gname = f"blk.{i}.attn_k.weight" if f"blk.{i}.attn_k.weight" in g.tensors else f"blk.{i}.attn_qkv.weight"
        hname = ckpt.find(layer_suffix(key["projection"]) + ".weight") if ckpt else None
        if gname in g.tensors and hname is not None:
            # Two heads tell the row order (and whether the weights are the same); the rest would only cost
            # downloads.
            heads = min(n_head_kv, 2)
            first = q_dim if key["projection_layout"].startswith("fused") else 0
            if key["projection_layout"].startswith("fused grouped"):
                q_per_group = n_head // n_head_kv
                hf = np.concatenate([ckpt.tensor(hname, (h * (q_per_group + 2) + q_per_group) * head_dim, head_dim)
                                     for h in range(heads)])
            else:
                hf = ckpt.tensor(hname, first, heads * head_dim)
            gg = g.values(gname, first, heads * head_dim)
            perm = rope_permutation(gg, hf, heads, head_dim) if hf.shape == gg.shape else None
            if perm is None:
                d.file_issues.append(f"{gname} does not hold the reference's {hname} in any row order: the GGUF "
                                     f"was converted from other weights")
            else:
                file_layout = pairing_in_file(ref_layout, perm[0], key["rope"]["dims"], head_dim)
                order = {"identity": "in the reference's order", "interleave": "interleaved per head (each "
                         "reference pair (i, i + d/2) stored at (2i, 2i + 1))",
                         "deinterleave": "de-interleaved per head"}[perm[0]]
                d.notes.append(f"{gname} holds {hname} {order} (relative error {perm[1]:.1e}, next order "
                               f"{perm[2]:.2f})")
                if file_layout is None:
                    d.missing.append("a RoPE pairing of dimensions that is neither (i, i + r/2) nor (2i, 2i + 1)")
        else:
            d.unknown.append(f"cannot compare the key projection of block {i}: "
                             + ("no reference weights" if hname is None else f"{gname} missing"))
    if file_layout is not None:
        d.set("rope", LAYOUT_ENUM[file_layout],
              f"the reference pairs ({'i, i + r/2' if ref_layout == 'halves' else '2i, 2i + 1'}) and the file "
              f"stores its key rows {'as is' if (file_layout == ref_layout) else 'reordered'}")

    # Frequencies: the engine's table from the file against the reference's, globally attending blocks first.
    windows = {b["index"]: b["mixer"].get("window") for b in att_blocks}
    global_tables = {b["mixer"]["key"]["rope"]["table"] for b in att_blocks
                     if b["mixer"].get("key") and b["mixer"]["key"].get("rope") and windows[b["index"]] is None}
    window_tables = {b["mixer"]["key"]["rope"]["table"] for b in att_blocks
                     if b["mixer"].get("key") and b["mixer"]["key"].get("rope") and windows[b["index"]] is not None}
    scaling_type = g.key("rope.scaling.type")
    if scaling_type not in (None, "none") and scaling_type not in engine.rope_scalings:
        d.missing.append(f"{'YaRN' if scaling_type == 'yarn' else repr(scaling_type)} RoPE scaling "
                         f"(rope.scaling.type {scaling_type})")
    _, _, qk_factor, _ = engine_inv_freq(g, rope_dim, int(g.key("context_length", 0) or 0))
    rope_dims = {b["mixer"]["key"]["rope"]["dims"] for b in att_blocks
                 if b["mixer"].get("key") and b["mixer"]["key"].get("rope")}
    if rope_dims and rope_dims != {rope_dim}:
        d.file_issues.append(f"the reference rotates {sorted(rope_dims)} dimensions per head, the file "
                             f"{rope_dim} (rope.dimension_count)")
    elif rope_dims:
        d.parts.append(f"RoPE over {rope_dim} of {head_dim} dimensions per head")

    def check_table(tname, label, rope_base=None):
        t = rope_tables[tname]
        ref = t["inv_freq"]
        if rope_base is None:
            inv, mag, _, how = engine_inv_freq(g, rope_dim, int(g.key("context_length", 0) or 0))
        else:
            inv = [float(np.float32(rope_base ** (-2.0 * i / rope_dim))) for i in range(rope_dim // 2)]
            mag, how = 1.0, f"base {rope_base:g}, unscaled"
        if len(inv) != len(ref) or not all(_close(a, b, 2e-5) for a, b in zip(inv, ref)):
            worst = max(range(min(len(inv), len(ref))), key=lambda i: abs(inv[i] - ref[i]) / max(ref[i], 1e-30))
            d.missing.append(f"{label}: the reference's RoPE frequencies are not the engine's from the file "
                             f"({how}): pair {worst} turns {ref[worst]:.6g} rad/position there, "
                             f"{inv[worst]:.6g} here" + (f" (rope type {cfg.get('rope_parameters', {}).get('rope_type')})"
                                                         if isinstance(cfg.get("rope_parameters"), dict) else ""))
            return False
        if not _close(mag, t["magnitude"], 1e-5):
            d.file_issues.append(f"{label}: RoPE magnitude {t['magnitude']:.6g} in the reference, "
                                 f"{mag:.6g} from the file (rope.scaling.attn_factor)")
            return False
        d.parts.append(f"{label}: RoPE frequencies from the file match the reference ({how})")
        return True

    # The table every block rotates with unless windowed blocks have one of their own (when every attention
    # block is windowed, theirs is that table).
    main_tables = global_tables or window_tables
    for tname in sorted(main_tables):
        check_table(tname, "attention blocks" if not (global_tables and window_tables) else "globally attending blocks")
    rope_params = cfg.get("rope_parameters") or {}
    rope_types = {v.get("rope_type") for v in rope_params.values() if isinstance(v, dict)} or {rope_params.get("rope_type")}
    if "dynamic" in rope_types:
        d.notes.append("the reference's dynamic RoPE recomputes its frequencies once a sequence grows past "
                       f"{cfg.get('max_position_embeddings')} positions; the engine keeps them")
    if "longrope" in rope_types:
        d.notes.append("LongRoPE: the engine picks the short or long factors once from the configured context; "
                       "the reference switches when a sequence passes the original context")

    # --- windows ---
    layer_types = cfg.get("layer_types")
    window_sizes = {w for w in windows.values() if w is not None}
    flags = None
    if layer_types and len(layer_types) == n_layers:
        traced_types = {}
        for b in bf:
            if b.get("mixer", {}).get("kind") == "attention":
                traced_types.setdefault(layer_types[b["index"]], set()).add(windows[b["index"]] is not None)
        if all(len(v) == 1 for v in traced_types.values()):
            flags = [next(iter(traced_types[t])) if t in traced_types else False for t in layer_types]
            missing_types = {t for t in layer_types if t not in traced_types and t != "linear_attention"}
            if missing_types:
                d.unknown.append(f"blocks of type {sorted(missing_types)} were not traced")
        else:
            d.unknown.append("blocks of the same layer type attend through different windows")
    elif att_blocks:
        traced = [windows[b["index"]] is not None for b in att_blocks]
        if all(traced):
            flags = [True] * n_layers
            d.notes.append(f"every traced attention block ({len(traced)} of {n_layers}) attends through the "
                           f"window and the configuration lists no per-block pattern: all blocks do")
        elif not any(traced):
            flags = [False] * n_layers
        else:
            d.unknown.append("some traced blocks attend through a window and the configuration lists no "
                             "per-block pattern (layer_types)")
    if flags is not None:
        for i, p in enumerate(plans):
            if p["mixer"] != "attention":
                flags[i] = False
        win = [i for i, f in enumerate(flags) if f]
        att_idx = [i for i, p in enumerate(plans) if p["mixer"] == "attention"]
        period = 0
        registered_default = engine.rows.get(arch, ({},))[0].get("sliding_window_default", 0)
        fallback = registered_default
        fw = g.key("attention.sliding_window")
        if win and fw is None and not fallback:
            d.unknown.append("the file omits its sliding window: a family-wide fallback cannot be established "
                             "from one checkpoint; record it in the file or establish the registry default")
        d.set("sliding_window_default", fallback, "registered family fallback checked against the reference when "
              "window metadata is absent" if fallback else "the file supplies the window, or the architecture has no window")
        if win:
            if len(window_sizes) != 1:
                d.unknown.append(f"windowed blocks use different windows: {sorted(window_sizes)}")
            if set(win) == set(att_idx):
                period = 1
            else:
                for n in range(2, n_layers + 1):
                    if all(flags[i] == ((i + 1) % n != 0) for i in att_idx):
                        period = n
                        break
                if period == 0:
                    pattern = g.key("attention.sliding_window_pattern")
                    if isinstance(pattern, list) and [bool(x) for x in pattern[:n_layers]] == flags:
                        period = 1
                        d.notes.append("irregular windowed blocks: the file's sliding_window_pattern lists them")
                    else:
                        d.missing.append(f"windowed blocks {_ranges(win)} follow no period and the file lists no "
                                         f"attention.sliding_window_pattern")
            w = next(iter(window_sizes)) if window_sizes else None
            if fw is None and fallback and w == fallback:
                d.parts.append(f"windowed blocks {_ranges(win)}: reference default {fallback}, omitted by the converter")
            elif fw is None:
                d.file_issues.append(f"the reference attends through a {w}-position window in blocks "
                                     f"{_ranges(win)}; the GGUF does not record it (attention.sliding_window)")
            elif w is not None and int(fw) != int(w):
                d.file_issues.append(f"window {w} in the reference, {fw} in the file")
            else:
                d.parts.append(f"windowed blocks {_ranges(win)}: the last {fw} positions")
        if period == 0:
            why = "no block attends through a window"
        elif period == 1 and set(win) == set(att_idx):
            why = "every attention block attends through the window"
        elif period == 1:
            why = "the file's sliding_window_pattern lists the windowed blocks"
        else:
            why = f"blocks {period - 1}, {2 * period - 1}, ... attend globally, the others through the window"
        d.set("swa_period", period, why)
        # Windowed blocks with their own RoPE table: an unscaled base of their own.
        base_swa = 0.0
        if win and global_tables and window_tables and window_tables != global_tables:
            if len(window_tables) != 1:
                d.unknown.append("windowed blocks rotate with several tables")
            else:
                t = rope_tables[next(iter(window_tables))]
                inv = t["inv_freq"]
                base = (1.0 / inv[1]) ** (rope_dim / 2.0) if len(inv) > 1 and inv[1] > 0 else None
                if base is None:
                    d.unknown.append("cannot read the windowed blocks' RoPE base")
                else:
                    base = float(f"{base:.6g}")
                    if check_table(t["name"], "windowed blocks", rope_base=base):
                        base_swa = base
                        fb = g.key("rope.freq_base_swa")
                        if fb is not None and not _close(float(fb), base):
                            d.file_issues.append(f"rope.freq_base_swa {fb} in the file, {base:g} in the reference")
        d.set("rope_base_swa", base_swa,
              f"windowed blocks rotate unscaled with base {base_swa:g}" if base_swa else
              "windowed blocks rotate like the others" if win else "no windowed block")

    # --- attention arguments ---
    qk_orders, qk_spans = set(), set()
    alibi = [b["mixer"].get("alibi_slopes") for b in att_blocks]
    if alibi and all(slopes is None for slopes in alibi):
        d.set("alibi_bias", 0.0, "no linear per-head position bias in the attention scores")
    elif alibi and all(slopes is not None for slopes in alibi):
        slopes = alibi[0]
        pow2 = 2 ** int(math.log2(len(slopes)))
        bias = float(f"{-math.log2(slopes[0]) * pow2:.6g}") if slopes[0] > 0 else None
        expected = [2 ** (-bias * (h + 1) / pow2) if h < pow2 else
                    2 ** (-bias * (2 * (h - pow2) + 1) / (2 * pow2)) for h in range(len(slopes))] if bias else []
        if bias and all(np.allclose(s, expected, rtol=1e-4, atol=1e-6) for s in alibi):
            d.set("alibi_bias", bias, "linear score slopes recomputed from native attention matmul and softmax")
            if g.key("attention.max_alibi_bias") is not None and not _close(
                    float(g.key("attention.max_alibi_bias")), bias, 1e-5):
                d.file_issues.append("attention.max_alibi_bias disagrees with the observed slopes")
        else:
            d.unknown.append("attention has linear position slopes that do not follow the ALiBi construction")
    elif alibi:
        d.missing.append("ALiBi in only some attention blocks")
    # The q.k scale the engine uses: the file's attention.scale when it reads that key, else 1/sqrt(head_dim);
    # times YaRN's q.k factor squared. Soft caps likewise come from the file when the engine reads them.
    file_scale = g.key("attention.scale") if "attention.scale" in engine.file_keys else None
    engine_scale = (float(file_scale) if file_scale is not None else _f32(1.0 / math.sqrt(head_dim))) * qk_factor ** 2
    file_cap = g.key("attn_logit_softcapping") if "attn_logit_softcapping" in engine.file_keys else None
    for b in att_blocks:
        m = b["mixer"]
        i = b["index"]
        if not _close(float(g.key("attention.clamp_kqv", 0.0)), float(m.get("clamp_kqv", 0.0)), 1e-5):
            d.file_issues.append(f"block {i}: attention.clamp_kqv differs from the observed symmetric clamp")
        if not _close(m["scale"], engine_scale, 1e-5):
            if "attention.scale" in engine.file_keys:
                d.file_issues.append(f"block {i}: attention scale {m['scale']:.6g} in the reference, "
                                     f"{engine_scale:.6g} from the file (attention.scale "
                                     f"{'absent' if file_scale is None else file_scale})")
            else:
                d.missing.append(f"block {i}: attention scale {m['scale']:.6g}, not 1/sqrt(head_dim) = "
                                 f"{m['head_dim'] ** -0.5:.6g}")
        elif not _close(engine_scale, head_dim ** -0.5, 1e-6):
            d.parts.append(f"attention scale {engine_scale:.6g} from the file, as the reference's")
        cap = m.get("softcap") or 0.0
        d.softcap = d.softcap or bool(cap)
        if "attn_logit_softcapping" not in engine.file_keys:
            if cap:
                d.missing.append(f"block {i}: attention logit soft-capping at {cap:g}")
        elif not _close(float(file_cap or 0.0), cap, 1e-6):
            d.file_issues.append(f"block {i}: attention soft-cap {cap:g} in the reference, {file_cap or 0:g} in the "
                                 f"file (attn_logit_softcapping)")
        elif cap:
            d.parts.append(f"attention scores soft-capped at {cap:g} (from the file)")
        if m.get("sinks"):
            d.missing.append(f"block {i}: attention sinks (a learned logit per head)")
        if not m.get("causal", True):
            d.missing.append(f"block {i}: non-causal attention")
        if m["head_dim"] != head_dim or m["heads"] != n_head or m["kv_heads"] != n_head_kv:
            d.file_issues.append(f"block {i}: {m['heads']}/{m['kv_heads']} heads of {m['head_dim']} in the "
                                 f"reference, {n_head}/{n_head_kv} of {head_dim} in the file")
        for side in ("key", "query"):
            info = m.get(side) or {}
            if info.get("norm"):
                qk_orders.add(info["norm"]["order"])
                qk_spans.add(info["norm"]["span"])
        gate = m.get("output_gate")
        plan = plans[i] if i < len(plans) else {}
        if bool(gate) != bool(plan.get("attn_gate")):
            d.file_issues.append(f"block {i}: the reference {'gates' if gate else 'does not gate'} the attention "
                                 f"output, the file's attn_q {'is not' if gate else 'is'} twice as tall")
        key = m.get("key") or {}
        if key.get("projection_layout", "").startswith("fused") != bool(plan.get("fused_qkv")):
            d.file_issues.append(f"block {i}: fused QKV in one of the reference and the file only")
    if qk_spans - {head_dim}:
        d.missing.append(f"Q/K norms over {sorted(qk_spans - {head_dim})} dimensions (all heads together), "
                         f"not each head's {head_dim}")
    if len(qk_orders) > 1:
        d.unknown.append("Q/K norms sit before RoPE in some blocks and after it in others")
    elif qk_orders:
        order = next(iter(qk_orders))
        d.set("qk_norm", QK_ORDER_ENUM[order], f"Q/K norms {order.replace('_', ' ')}")
    elif att_blocks:
        d.set("qk_norm", engine_default(engine, "qk_norm"), "no Q/K norms (the default)")

    # --- norms: placement and formula, and which GGUF tensor holds each ---
    roles = ("norm_before_mixer", "norm_after_mixer", "norm_before_ffn", "norm_after_ffn")
    placement = {}
    for r in roles:
        present = {b.get(r) is not None for b in bf if "mixer" in b}
        if len(present) > 1:
            d.unknown.append(f"{r.replace('_', ' ')} exists in some blocks only")
        placement[r] = present == {True}
    formulas = {(b.get(r) or {}).get("formula") for b in bf for r in roles if b.get(r)}
    if facts.get("final_norm"):
        formulas.add(facts["final_norm"]["formula"])
    if None in formulas:
        d.unknown.append("a norm whose formula is not RMSNorm (x w or x (1 + w)) or LayerNorm")
    norm_kinds = {"Layer" if f in ("layer", "layer_1p") else "Rms" for f in formulas if f is not None}
    if len(norm_kinds) == 1:
        d.set("norm", next(iter(norm_kinds)), "norm formula recomputed from every block and the final norm")
    elif norm_kinds:
        d.missing.append("different norm formulas across the block and output norms")
    if "rms_noweight" in formulas:
        d.missing.append("RMSNorm without a weight")
    eps_ref = {(b.get(r) or {}).get("eps") for b in bf for r in roles if b.get(r)} - {None}
    if eps_ref and not all(_close(e, eps, 1e-3) for e in eps_ref):
        d.file_issues.append(f"norm epsilon {sorted(eps_ref)} in the reference, {eps} in the file")
    if any(isinstance(b.get("residual"), dict) for b in bf):
        mults = sorted({round(b["residual"]["multiplier"], 6) for b in bf if isinstance(b.get("residual"), dict)})
        d.missing.append(f"residual multiplier {mults}: the block's mixer and FFN outputs are scaled before "
                         f"they are added")

    norm_names = {}
    b0 = next((b for b in bf if "mixer" in b and b["index"] < n_layers), None)
    if b0 is not None and ckpt is not None:
        i = b0["index"]
        candidates = {s: g.values(f"blk.{i}.{s}").reshape(-1) for s in blocks.get(i, set())
                      if s.endswith(".weight") and len(g.tensors[f"blk.{i}.{s}"]["dims"]) == 1
                      and g.tensors[f"blk.{i}.{s}"]["dims"][0] == n_embd}
        for r in roles:
            info = b0.get(r)
            if info is None:
                continue
            hname = ckpt.find(layer_suffix(info["module"]) + ".weight")
            if hname is None:
                if info.get("weighted") is False:
                    continue  # non-parametric LayerNorm has no stored tensor to match
                d.unknown.append(f"no reference weights for {info['module']}")
                continue
            w = ckpt.tensor(hname).reshape(-1)
            want = w + 1.0 if info["formula"] in ("rms_1p", "layer_1p") else w
            hits = [s for s, v in candidates.items() if v.shape == want.shape and _rel_err(v, want) < 1e-2]
            if len(hits) == 1:
                norm_names[r] = hits[0][: -len(".weight")]
            elif not hits:
                stored_raw = [s for s, v in candidates.items() if v.shape == w.shape and _rel_err(v, w) < 1e-2]
                if info["formula"] == "rms_1p" and stored_raw:
                    d.file_issues.append(f"blk.{i}.{stored_raw[0]} stores w where the reference multiplies by "
                                         f"1 + w ({hname})")
                else:
                    d.file_issues.append(f"no tensor of block {i} holds the reference's {hname}")
            else:
                d.unknown.append(f"several tensors of block {i} hold the reference's {hname}: {hits}")
    elif b0 is not None:
        d.unknown.append("cannot tell which GGUF tensor holds which norm without the reference weights")
    if placement.get("norm_before_mixer") and norm_names.get("norm_before_mixer") not in (None, "attn_norm"):
        d.missing.append(f"the norm before the mixer is {norm_names['norm_before_mixer']}, the engine binds attn_norm")
    if placement.get("norm_before_ffn"):
        if "norm_before_ffn" in norm_names:
            d.set("ffn_norm", norm_names["norm_before_ffn"],
                  f"{b0['norm_before_ffn']['module'].rsplit('.', 1)[-1]} normalizes the residual stream before the "
                  f"FFN; the GGUF holds its weights as blk.N.{norm_names['norm_before_ffn']}")
        elif b0 and (b0.get("norm_before_ffn") or {}).get("weighted") is False:
            d.set("ffn_norm", "ffn_norm", "the FFN's norm has no parameters (optional tensor name is unused)")
    elif b0 and b0.get("parallel"):
        existing = engine.rows.get(g.arch)
        # An absent FFN norm has no name to establish. Keep the registered
        # optional name if it is absent from the file, else use ffn_norm.
        name = existing[0].get("ffn_norm") if existing else "ffn_norm"
        if any(f"{name}.weight" in ts or f"{name}.bias" in ts for ts in blocks.values()):
            d.file_issues.append(f"the FFN shares its attention norm, but the file has an extra {name}")
        else:
            d.set("ffn_norm", name, "the parallel FFN shares the mixer's norm (no FFN norm tensors)")
    parallels = {bool(b.get("parallel")) for b in bf if "mixer" in b}
    if len(parallels) == 1:
        parallel = next(iter(parallels))
        flag = g.key("use_parallel_residual")
        if flag is not None and bool(flag) != parallel:
            d.file_issues.append("use_parallel_residual disagrees with the observed residual inputs")
        # If metadata states it, the registry default is unused.
        d.set("parallel_residual", parallel,
              "attention and FFN read the block input" if parallel else "the FFN follows the attention residual")
    elif parallels:
        d.missing.append("parallel residuals in only some blocks")
    for r, field in (("norm_after_mixer", "post_mixer_norm"), ("norm_after_ffn", "post_ffn_norm")):
        if not placement.get(r):
            d.set(field, "", f"no norm after the {'mixer' if 'mixer' in r else 'FFN'}")
        elif r in norm_names:
            d.set(field, norm_names[r],
                  f"{b0[r]['module'].rsplit('.', 1)[-1]} normalizes the {'mixer' if 'mixer' in r else 'FFN'}'s "
                  f"output before the residual add; GGUF blk.N.{norm_names[r]}")
    # Q/K norms: their formula, and whether the file stores what the engine multiplies by.
    for b in att_blocks[:1]:
        for side, gsuffix in (("key", "attn_k_norm.weight"), ("query", "attn_q_norm.weight")):
            norm = (b["mixer"].get(side) or {}).get("norm")
            if not norm:
                continue
            if norm.get("formula") not in ("rms", "rms_1p", "layer", "layer_1p"):
                d.unknown.append(f"block {b['index']}: the {side} norm has an unknown formula")
                continue
            gname = f"blk.{b['index']}.{gsuffix}"
            hnames = [ckpt.find(layer_suffix(module) + ".weight") for module in norm.get("modules", [norm["module"]])] if ckpt else []
            hname = hnames[0] if hnames and all(hnames) else None
            if gname not in g.tensors:
                d.file_issues.append(f"the reference normalizes {side}s, the GGUF has no {gname}")
            elif hname is not None:
                w = np.concatenate([ckpt.tensor(name).reshape(-1) for name in hnames])
                want = w + 1.0 if norm["formula"] == "rms_1p" else w
                have = g.values(gname).reshape(-1)
                if have.shape != want.shape or _rel_err(have, want) > 1e-2:
                    d.file_issues.append(f"{gname} does not hold the reference's {hname}"
                                         + (" as 1 + w" if norm["formula"] == "rms_1p" else ""))
    if facts.get("final_norm") and ckpt is not None and "output_norm.weight" in g.tensors:
        fn = facts["final_norm"]
        hname = ckpt.find(fn["module"].split(".", 1)[-1] + ".weight") or ckpt.find(fn["module"] + ".weight")
        if hname is not None:
            w = ckpt.tensor(hname).reshape(-1)
            want = w + 1.0 if fn["formula"] in ("rms_1p", "layer_1p") else w
            if _rel_err(g.values("output_norm.weight").reshape(-1), want) > 1e-2:
                d.file_issues.append(f"output_norm.weight does not hold the reference's {hname}"
                                     + (" as 1 + w" if fn["formula"] == "rms_1p" else ""))

    # --- FFN ---
    acts = set()
    moe_facts = [b["ffn"] for b in bf if (b.get("ffn") or {}).get("kind") == "moe"]
    for b in bf:
        f = b.get("ffn") or {}
        i = b["index"]
        plan = plans[i] if i < len(plans) else {}
        for u in f.get("unknown", []):
            d.unknown.append(f"block {i}: {u}")
        if f.get("kind") == "dense":
            if f.get("gated") is False:
                ts = blocks.get(i, set())
                if "ffn_gate.weight" in ts or suffix_rows(i, "ffn_up.weight") != n_ff:
                    d.file_issues.append(f"block {i}: ungated FFN in the reference but not in the file")
            if f.get("activation"):
                acts.add(f["activation"])
            if f.get("activation") == "xielu":
                for name, traced_value in f["scalars"].items():
                    hname = ckpt.find(layer_suffix(f["activation_module"]) + "." + name) if ckpt else None
                    actual = float(ckpt.tensor(hname).reshape(-1)[0]) if hname else traced_value
                    values = g.metadata.get("xielu." + name)
                    if values is None or len(values) <= i or not _close(float(values[i]), actual, 1e-5):
                        d.file_issues.append(f"xielu.{name} does not hold the reference's block {i} scalar")
            if f.get("fused_gate_up") and f.get("fused_order") != "gate_first":
                d.missing.append(f"block {i}: a fused [up | gate] projection (the engine reads [gate | up])")
            if plan and plan.get("ffn") != "dense":
                d.file_issues.append(f"block {i}: dense FFN in the reference, MoE in the file")
            elif plan and bool(f.get("fused_gate_up")) != bool(plan.get("fused_gate_up")):
                d.file_issues.append(f"block {i}: fused gate/up projection in one of the reference and the file only")
        elif f.get("kind") == "moe":
            if f.get("expert_activation"):
                acts.add(f["expert_activation"])
            if plan and plan.get("ffn") != "moe":
                d.file_issues.append(f"block {i}: MoE in the reference, dense FFN in the file")
                continue
            if plan.get("fused_experts") and "ffn_gate_up_exps.weight" not in engine.block_tensors:
                d.missing.append(f"block {i}: fused ffn_gate_up_exps expert tensors")
            if not f.get("gating"):
                continue
            gating = {1: "softmax", 2: "sigmoid"}.get(int(g.key("expert_gating_func", 1) or 1))
            if gating != f["gating"]:
                d.file_issues.append(f"block {i}: {f['gating']} routing in the reference, {gating} from the file "
                                     f"(expert_gating_func)")
            if bool(f.get("selection_bias")) != bool(plan.get("selection_bias")):
                d.file_issues.append(f"block {i}: an expert selection bias in one of the reference and the file only")
            fnorm = g.key("expert_weights_norm")
            if fnorm is not None and bool(fnorm) != bool(f["norm_topk"]):
                d.file_issues.append(f"block {i}: expert_weights_norm {fnorm} in the file, the reference "
                                     f"{'renormalizes' if f['norm_topk'] else 'does not renormalize'}")
            fscale = float(g.key("expert_weights_scale", 1.0) or 1.0)
            if not _close(fscale, f["scale"], 1e-4):
                d.file_issues.append(f"block {i}: expert weights scaled by {f['scale']:g} in the reference, "
                                     f"{fscale:g} from the file")
            sh = f.get("shared_expert")
            if bool(sh) != bool(plan.get("shared_expert")):
                d.file_issues.append(f"block {i}: a shared expert in one of the reference and the file only")
            elif sh and bool(sh.get("gate")) != bool(plan.get("shared_gate")):
                d.file_issues.append(f"block {i}: the shared expert's gate in one of the reference and the file only")
    if len(acts) > 1:
        d.missing.append(f"FFNs with different activations {sorted(acts)} (one activation per architecture)")
    elif acts:
        act = acts.pop()
        enum = ACTIVATION_ENUM.get(act)
        if enum is None or enum not in engine.enums.get("FfnActivation", []):
            d.missing.append(f"FFN activation {act} (FfnActivation has {engine.enums.get('FfnActivation')})")
        else:
            gated = bool(moe_facts) or any((b.get("ffn") or {}).get("gated") for b in bf)
            d.set("ffn_activation", enum, f"the {'experts compute' if moe_facts else 'FFN computes'} "
                  + (f"down({act}(gate) * up)" if gated else f"down({act}(up))"))
    if moe_facts:
        norms_ = {f.get("norm_topk") for f in moe_facts if "norm_topk" in f}
        if len(norms_) == 1:
            v = norms_.pop()
            d.set("moe_norm_weights", bool(v), f"the router {'renormalizes' if v else 'does not renormalize'} "
                                                f"the top-k weights")
        if any(f.get("gating") for f in moe_facts):
            f = next(f for f in moe_facts if f.get("gating"))
            d.parts.append(f"MoE: {f['gating']} routing, top {g.key('expert_used_count')} of "
                           f"{g.key('expert_count')} experts" + (", selection bias" if f.get("selection_bias") else "")
                           + (f", weights scaled by {f['scale']:g}" if not _close(f['scale'], 1.0) else "")
                           + (", a shared expert" + (" with a sigmoid gate" if f["shared_expert"].get("gate") else "")
                              if f.get("shared_expert") else ""))
    elif bf:
        d.set("moe_norm_weights", False, "dense FFNs (the MoE default does not apply)")

    # --- embeddings and logits ---
    pos = facts.get("position_embeddings")
    if pos:
        tensor = g.tensors.get("position_embd.weight")
        if tensor is None or tuple(tensor["dims"]) != (n_embd, pos["rows"]):
            d.file_issues.append("position_embd.weight does not hold the reference's learned position table")
        elif ckpt is not None:
            hname = ckpt.find(pos["module"] + ".weight")
            if hname and _rel_err(g.values("position_embd.weight"), ckpt.tensor(hname)) > 1e-2:
                d.file_issues.append("position_embd.weight differs from the reference weights")
        d.parts.append(f"learned absolute positions: {pos['rows']} rows, added before block 0")
    elif "position_embd.weight" in g.tensors:
        d.file_issues.append("the file has learned position embeddings absent from the observed input")
    emb_norm = facts.get("embedding_norm")
    if emb_norm:
        if "token_embd_norm.weight" not in g.tensors:
            d.file_issues.append("the reference normalizes token embeddings but the file has no token_embd_norm.weight")
        if (emb_norm.get("formula") in ("layer", "layer_1p")) != (d.fields.get("norm", (None,))[0] == "Layer"):
            d.missing.append("the embedding norm uses a different formula from the block norms")
        d.parts.append("token embeddings normalized before block 0")
    es = facts.get("embedding_scale")
    if es is None:
        pass  # reported as unknown by the trace
    elif _close(es, 1.0, 1e-6):
        d.set("scale_embeddings", False, "block 0 receives the embedding rows as they are")
    elif _close(es, float(np.float32(math.sqrt(n_embd))), 1e-5):
        d.set("scale_embeddings", True, f"block 0 receives the embedding rows times sqrt({n_embd}) = {es:.6g}")
    else:
        d.missing.append(f"embeddings multiplied by {es:g} (the engine scales by sqrt(n_embd) = "
                         f"{math.sqrt(n_embd):.6g} or not at all)")
    lg = facts.get("logits") or {}
    final_cap = lg["value"] if lg.get("transform") == "softcap" else 0.0
    if lg.get("transform") == "scale":
        if not _close(float(g.key("logit_scale", 1.0)), lg["value"], 1e-5):
            d.file_issues.append(f"logit_scale {g.key('logit_scale', 1.0)} in the file, {lg['value']:.6g} in the reference")
        else:
            d.parts.append(f"logits multiplied by {lg['value']:.6g} from the file")
    elif "final_logit_softcapping" not in engine.file_keys:
        if final_cap:
            d.missing.append(f"final logit soft-capping at {final_cap:g}")
    elif not _close(float(g.key("final_logit_softcapping", 0.0) or 0.0), final_cap, 1e-6):
        d.file_issues.append(f"final logits soft-capped at {final_cap:g} in the reference, "
                             f"{g.key('final_logit_softcapping', 0)} in the file (final_logit_softcapping)")
    elif final_cap:
        d.parts.append(f"final logits soft-capped at {final_cap:g} (from the file)")
    tied = facts.get("tied_output")
    has_output = "output.weight" in g.tensors
    if not has_output and tied is False:
        d.file_issues.append("the reference has its own output matrix, the GGUF has no output.weight")
    else:
        d.parts.append("output matrix " + ("output.weight" if has_output else "tied to token_embd"))

    check_tokenizer_settings(d, g, model, revision, trust_remote_code, engine)

    # --- the row: values for every field ---
    row = {}
    for name, ftype, default in engine.fields:
        if name in d.fields:
            row[name] = d.fields[name][0]
        elif default is not None and name in DERIVED_FIELDS:
            row[name] = default
    complete = all(n in row for n, _, _ in engine.fields)
    # A required field the reference has no counterpart for is a missing part (reported above), not an unknown.
    no_ffn_norm = any("no norm before the FFN" in m for m in d.missing)
    for n in ("name", "rope", "ffn_norm", "moe_norm_weights"):
        if n not in d.fields and n in [f for f, _, _ in engine.fields] and not (n == "ffn_norm" and no_ffn_norm):
            d.unknown.append(f"{n}: not established")

    # --- every tensor must be bound (the norms' names come from the row, so only once they are known) ---
    norm_fields = ("ffn_norm", "post_mixer_norm", "post_ffn_norm")
    if no_ffn_norm:
        pass  # the block's structure is already not the engine's
    elif all(f in d.fields for f in norm_fields):
        names = {row.get(f) for f in norm_fields if row.get(f)}
        bound = engine.block_tensors | {n + suffix for n in names for suffix in (".weight", ".bias")}
        unused = {}
        for b in range(n_layers):
            for s in blocks.get(b, set()) - bound:
                unused.setdefault(s, []).append(b)
        for s, bl in sorted(unused.items()):
            d.missing.append(f"tensor {s} (blocks {_ranges(bl)}) is not used by the engine")
    else:
        d.unknown.append("which block tensors the engine would bind (the norms' names are not established)")
    for name in sorted(globals_ - engine.global_tensors):
        d.missing.append(f"tensor {name} is not used by the engine")
    rec = [i for i, p in enumerate(plans) if p["mixer"] == "deltanet"]
    for i in rec:
        lacking = DELTANET_TENSORS - blocks.get(i, set())
        if lacking:
            d.missing.append(f"block {i}: a recurrent mixer without the Gated DeltaNet tensors {sorted(lacking)}")
            break
    if rec and rec_blocks:
        d.notes.append(f"recurrent blocks: the reference runs {', '.join(rec_blocks[0]['mixer']['classes'])}; "
                       f"the file holds the engine's Gated DeltaNet tensors")

    # --- the per-block plan, grouped ---
    groups = {}
    for i, p in enumerate(plans):
        if p["mixer"] == "deltanet":
            mix = "Gated DeltaNet"
        else:
            mix = "attention" + ("".join(f", {t}" for t, on in (("fused QKV", p["fused_qkv"]),
                                                                ("QKV biases", p["qkv_bias"]),
                                                                ("Q/K norms", p["qk_norm"]),
                                                                ("gated output", p["attn_gate"])) if on))
        if p["ffn"] == "moe":
            ffn = "MoE FFN" + (" + shared expert" if p["shared_expert"] else "")
        else:
            ffn = "dense FFN" + (" (fused gate/up)" if p["fused_gate_up"] else "")
        groups.setdefault(f"{mix}; {ffn}", []).append(i)
    for what, idx in groups.items():
        d.parts.append(f"{'block' if len(idx) == 1 else 'blocks'} {_ranges(idx)}: {what}")

    # --- tokenizer ---
    kind = g.metadata.get("tokenizer.ggml.model")
    pre = g.metadata.get("tokenizer.ggml.pre", "default")
    if kind not in engine.tokenizer_kinds:
        d.missing.append(f"tokenizer type '{kind}' (the engine reads {engine.tokenizer_kinds})")
    elif kind == "gpt2" and pre not in engine.pre_tokenizers:
        d.missing.append(f"BPE pre-tokenizer '{pre}' (tokenizer.ggml.pre) is not in the pre-tokenizer table")
    else:
        d.parts.append(f"tokenizer {kind}" + (f", pre-tokenizer {pre}" if kind == "gpt2" else ""))

    d.merge_blocks()
    if complete and not d.unknown:
        d.row = engine.format_row(row)
    if arch in engine.rows:
        values, _ = engine.rows[arch]
        diff = {n: (values.get(n), row.get(n)) for n in DERIVED_FIELDS if n in values and values.get(n) != row.get(n)}
        d.registry = {"exists": True, "identical": complete and not diff and not d.unknown,
                      "differences": {k: {"registry": a, "derived": b} for k, (a, b) in diff.items()}}
        if diff and complete:
            d.notes.append("the registry's row for " + arch + " differs: " + ", ".join(
                f"{k} {a!r} there, {b!r} here" for k, (a, b) in diff.items()))
    else:
        d.registry = {"exists": False}
    return d, row if complete else None, g


def engine_default(engine, field):
    return next(dflt for n, _, dflt in engine.fields if n == field)


def check_tokenizer_settings(d, g, model, revision, trust_remote_code, engine):
    """The GGUF's BOS and space-prefix rules against what the model's own tokenizer does (the engine follows the
    file: tokenizer.ggml.add_bos_token, by default on for SentencePiece only; tokenizer.ggml.add_space_prefix, by
    default on). A file that disagrees encodes every prompt differently from the model's training."""
    kind = g.metadata.get("tokenizer.ggml.model")
    if kind != "llama":
        d.set("spm_prefix_after_control", True, "not a SentencePiece vocabulary; control-prefix policy is unused")
        d.set("spm_add_bos_default", True, "not a SentencePiece vocabulary; absent BOS metadata means no BOS")
        d.set("spm_add_space_prefix_default", True, "not a SentencePiece vocabulary; dummy-space policy is unused")
        d.set("spm_charsmap", "", "not a SentencePiece vocabulary; compiled character map is unused")
        d.set("spm_user_defined", "", "not a SentencePiece vocabulary; user-defined symbol fallback is unused")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=trust_remote_code)
    except Exception as e:  # noqa: BLE001 - reported, the gate checks the ids anyway
        d.notes.append(f"the reference tokenizer did not load ({type(e).__name__}); its BOS and space-prefix rules "
                       f"are left to the gate")
        if kind == "llama":
            d.unknown.append("spm_prefix_after_control: the reference tokenizer could not be loaded")
            d.unknown.append("spm_add_bos_default: the reference tokenizer could not be loaded")
        return
    bos = g.metadata.get("tokenizer.ggml.bos_token_id")
    default_bos = kind == "llama" and engine.rows.get(g.arch, ({},))[0].get("spm_add_bos_default", True)
    add_bos = g.metadata.get("tokenizer.ggml.add_bos_token", default_bos)
    add_bos = bool(add_bos) and bos is not None
    plain = tok("The sea", add_special_tokens=False)["input_ids"]
    special = tok("The sea", add_special_tokens=True)["input_ids"]
    ref_bos = len(special) > len(plain) and special[0] == bos and special[1:1 + len(plain)] == plain
    if kind == "llama":
        d.set("spm_add_bos_default", ref_bos, "observed initial BOS behavior of the reference tokenizer")
    # What the model's tokenizer_config.json declares. transformers does not always apply it (version 5 runs
    # DeepSeek's tokenizer.json as it is, without the BOS its LlamaTokenizerFast class added).
    declared = None
    try:
        path = os.path.join(model, "tokenizer_config.json") if os.path.isdir(model) else None
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model, "tokenizer_config.json", revision=revision)
        declared = json.load(open(path, encoding="utf-8")).get("add_bos_token")
    except Exception:  # noqa: BLE001 - no declaration: the behaviour decides
        pass
    gguf_rule = f"tokenizer.ggml.add_bos_token {g.metadata.get('tokenizer.ggml.add_bos_token', 'absent')}"
    if add_bos != ref_bos:
        if isinstance(declared, bool) and declared == add_bos:
            d.notes.append(f"BOS: tokenizer_config.json declares add_bos_token {declared} and the GGUF follows it "
                           f"({gguf_rule}), but transformers' tokenizer {'adds' if ref_bos else 'adds no'} BOS; the "
                           f"gate, whose reference is transformers', sees the first token differ")
        else:
            d.file_issues.append(f"the model's tokenizer {'adds' if ref_bos else 'adds no'} BOS, the GGUF "
                                 f"{'asks for one' if add_bos else 'asks for none'} ({gguf_rule})")
    if kind == "llama" and plain:
        piece = tok.convert_ids_to_tokens(plain[0])
        ref_prefix = isinstance(piece, str) and piece.startswith("▁")
        d.set("spm_add_space_prefix_default", ref_prefix, "observed initial dummy-space behavior of the reference tokenizer")
        default_prefix = engine.rows.get(g.arch, ({},))[0].get("spm_add_space_prefix_default", True)
        add_prefix = bool(g.metadata.get("tokenizer.ggml.add_space_prefix", default_prefix))
        if add_prefix != ref_prefix:
            d.file_issues.append(f"the model's tokenizer {'puts a' if ref_prefix else 'puts no'} space before the "
                                 f"text, the GGUF {'asks for one' if add_prefix else 'asks for none'} "
                                 f"(tokenizer.ggml.add_space_prefix "
                                 f"{g.metadata.get('tokenizer.ggml.add_space_prefix', 'absent')})")
        # GGUF does not preserve Metaspace's prepend_scheme. Observe a plain
        # segment after a literal control, independently of the initial BOS.
        policies = set()
        for token in (tok.bos_token, tok.eos_token):
            if not token:
                continue
            control_id = tok.convert_tokens_to_ids(token)
            ids = tok("hello" + token + "world", add_special_tokens=False)["input_ids"]
            if control_id not in ids:
                continue
            index = ids.index(control_id) + 1
            if index < len(ids):
                piece = tok.convert_ids_to_tokens(ids[index])
                if isinstance(piece, str):
                    policies.add(piece.startswith("▁"))
        if len(policies) == 1:
            d.set("spm_prefix_after_control", policies.pop(),
                  "observed dummy-space behavior after a literal BOS/EOS control token")
        else:
            d.unknown.append("spm_prefix_after_control: no consistent control-prefix behavior was observed")
        sp_model = getattr(tok, "sp_model", None)
        if sp_model is not None and hasattr(sp_model, "serialized_model_proto"):
            from spm_normalizer import read_normalizer
            normalizer = read_normalizer(sp_model.serialized_model_proto())
            digest = normalizer["sha256"]
            if normalizer["remove_extra_whitespaces"] or not normalizer["escape_whitespaces"]:
                d.missing.append("SentencePiece normalization with collapsed or unescaped whitespace")
            explicit = g.metadata.get("tokenizer.ggml.precompiled_charsmap")
            if explicit is not None:
                import hashlib
                if hashlib.sha256(bytes(explicit)).hexdigest() != digest:
                    d.file_issues.append("the GGUF precompiled character map differs from the reference")
                fallback = engine.rows.get(g.arch, ({},))[0].get("spm_charsmap", "")
            else:
                fallback = digest
                with open(os.path.join(REPO, "src/core/spm_normalizer.inc"), encoding="utf-8") as source:
                    registered = source.read(400)
                if digest and digest not in registered:
                    d.missing.append("the file omits its SentencePiece character map and no exact registered map matches")
            d.set("spm_charsmap", fallback, "exact SHA-256 of the native SentencePiece normalization map")
            symbols = normalizer["user_defined"]
            vocabulary = g.metadata.get("tokenizer.ggml.tokens", [])
            types = g.metadata.get("tokenizer.ggml.token_type", [])
            ids = {token: i for i, token in enumerate(vocabulary)}
            preserved = all(token in ids and ids[token] < len(types) and types[ids[token]] == 4 for token in symbols)
            symbols_fallback = engine.rows.get(g.arch, ({},))[0].get("spm_user_defined", "") if preserved else normalizer["user_defined_sha256"]
            if symbols_fallback:
                with open(os.path.join(REPO, "src/core/spm_normalizer.inc"), encoding="utf-8") as source:
                    if symbols_fallback not in source.read():
                        d.missing.append("the file loses SentencePiece user-defined types and no exact registered symbols match")
            d.set("spm_user_defined", symbols_fallback, "exact native user-defined symbols, or the file's preserved token types")
        else:
            # Fast Metaspace tokenizers with no compiled map are checked by
            # the token gate; the existing normalizer profiles own their NFC.
            d.set("spm_charsmap", "", "the reference tokenizer has no native compiled SentencePiece map")
            d.set("spm_user_defined", "", "the fast tokenizer's user-defined symbols come from the GGUF token types")


# ----------------------------------------------------------------------------------------------------------------
# The conformance gate
# ----------------------------------------------------------------------------------------------------------------


def run(cmd, log, cwd=None, env=None):
    """Runs cmd with its output appended to log; the console shows the programs by name, the log in full."""
    shown = [os.path.basename(c) if os.path.isabs(c) else c for c in cmd]
    print("  $ " + " ".join(shown[1:] if cmd[0] == sys.executable else shown), flush=True)
    with open(log, "a") as out:
        out.write("$ " + " ".join(cmd) + "\n")
        out.flush()
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace")
        out.write(p.stdout)
    return p.returncode, p.stdout


def sync_tree(src, dst):
    """Copies the source tree (tracked files and untracked ones git does not ignore) to dst, leaving files whose
    contents did not change untouched so that the scratch build stays incremental."""
    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=src, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    if listed.returncode == 0:
        files = listed.stdout.split("\n")
    else:  # not a git checkout: every file but build outputs
        files = []
        for root, dirs, names in os.walk(src):
            dirs[:] = [x for x in dirs if not x.startswith(("build", ".")) and x != "__pycache__"]
            files += [os.path.relpath(os.path.join(root, n), src) for n in names]
    wanted = set()
    for f in files:
        if not f or f.startswith(("app/", "docs/")) or not os.path.isfile(os.path.join(src, f)):
            continue
        wanted.add(f)
        a, b = os.path.join(src, f), os.path.join(dst, f)
        if os.path.exists(b) and os.path.getsize(a) == os.path.getsize(b):
            with open(a, "rb") as x, open(b, "rb") as y:
                if x.read() == y.read():
                    continue
        os.makedirs(os.path.dirname(b), exist_ok=True)
        shutil.copy2(a, b)
    for root, _, names in os.walk(dst):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), dst)
            if rel not in wanted:
                os.remove(os.path.join(root, n))


def patch_registry(text, arch, row_text, replace):
    """architecture.cpp with `row_text` added to kArchitectures (or replacing the row named `arch`)."""
    m = re.search(r"(kArchitectures\[\]\s*=\s*\{)(.*?)(\n\};)", text, re.S)
    body = m.group(2)
    if replace:
        entries = Engine._entries(body)
        for e in entries:
            if re.search(r'(^\{|\.name\s*=\s*)"' + re.escape(arch) + '"', e.strip()):
                start = body.index(e)
                end = start + len(e)
                while end < len(body) and body[end] in ",":
                    end += 1
                line_start = body.rfind("\n", 0, start) + 1
                body = body[:line_start] + row_text + body[end:]
                break
        else:
            raise ValueError(f"no row named {arch}")
    else:
        body = body.rstrip() + "\n" + row_text
    return text[:m.start(2)] + body + text[m.end(2):]


def gate(args, d, row_text, arch, engine):
    """Builds liyab-kl and liyab-tok with the row applied in a scratch copy and runs the conformance checks."""
    scratch = os.path.abspath(args.scratch)
    src = os.path.join(scratch, "src")
    build = os.path.join(scratch, "build")
    work = os.path.join(scratch, "runs", re.sub(r"[^A-Za-z0-9_.-]", "_", args.model))
    os.makedirs(src, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    log = os.path.join(work, "gate.log")
    open(log, "w").close()
    results = {"log": log, "checks": []}
    print(f"\nConformance gate (scratch tree {scratch}, log {log})", flush=True)
    sync_tree(args.source, src)
    target = os.path.join(src, "src/core/architecture.cpp")
    original = open(target, encoding="utf-8").read()
    exists = arch in engine.rows
    same = exists and engine.rows[arch][0] == engine.parse_row(Engine._entries(row_text)[0])
    patched = original if same else patch_registry(original, arch, row_text, replace=exists)
    if not same:
        with open(target, "w", encoding="utf-8") as f:
            f.write(patched)
    print("  registry: " + ("the existing row is the derived row; no patch" if same else
                            ("replacing" if exists else "adding") + f" the row for {arch}"), flush=True)
    try:
        if not os.path.exists(os.path.join(build, "build.ninja")):
            code, _ = run(["cmake", "-S", src, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                           "-DLIYAB_BUILD_TESTS=ON", "-DLIYAB_BUILD_CLI=OFF", "-DLIYAB_USE_METAL=OFF"], log)
            if code != 0:
                results["checks"].append({"check": "configure", "pass": False})
                return False, results
        code, _ = run(["cmake", "--build", build, "--target", "liyab-kl", "liyab-tok"], log)
        if code != 0:
            results["checks"].append({"check": "build", "pass": False})
            return False, results
    finally:
        if not same:  # revert: the scratch tree stays a copy of the source
            with open(target, "w", encoding="utf-8") as f:
                f.write(original)
    py = sys.executable
    tools = os.path.join(args.source, "tools")
    texts = [os.path.join(args.source, "tests/data", t) for t in ("conform_text.txt", "tokenizer_edge.txt",
                                                                   "tokenizer_chat.txt")]
    ok = True
    for text in texts:
        ids = os.path.join(work, os.path.basename(text) + ".ids")
        cmd = [py, os.path.join(tools, "tokenizer_reference.py"), "--model", args.model, "--text", text, "--out", ids]
        if args.revision:
            cmd += ["--revision", args.revision]
        if not has_tokenizer_json(args.model, args.revision):
            cmd += ["--transformers"] + (["--trust-remote-code"] if args.trust_remote_code else [])
        code, out = run(cmd, log)
        if code == 0:
            code, out = run([os.path.join(build, "liyab-tok"), "-m", args.gguf, "-f", text, "--ids", ids, "--quiet"],
                            log)
        # liyab-tok's verdict line: "same ids: N tokens" or "tokens: ... first difference at token K".
        lines_ = out.strip().splitlines()
        last = next((ln for ln in lines_ if ln.startswith(("same ids", "tokens:"))), lines_[-1] if lines_ else "")
        results["checks"].append({"check": f"tokenizer {os.path.basename(text)}", "pass": code == 0, "detail": last})
        print(f"  tokenizer, {os.path.basename(text)}: {'pass' if code == 0 else 'FAIL'} ({last})", flush=True)
        ok = ok and code == 0
    ref = os.path.join(work, "reference.kl")
    conform = os.path.join(tools, "conform_reference.py")
    cmd = [py, conform, "--model", args.model, "--text", texts[0], "--out", ref]
    if args.revision:
        cmd += ["--revision", args.revision]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    # transformers' default attention kernel leaves a soft cap out: the reference then needs the eager one.
    if d.softcap and "--attn-implementation" in open(conform, encoding="utf-8").read():
        cmd += ["--attn-implementation", "eager"]
    code, out = run(cmd, log)
    if code == 0:
        # On the reference's ids, so that the forward pass (the row) is measured even when this file encodes
        # the text differently; liyab-kl still reports whether its own encoding, BOS rule included, matches.
        code, out = run([os.path.join(build, "liyab-kl"), "-m", args.gguf, "-f", texts[0], "--compare", ref,
                         "--ref-tokens", "--kv", "f16", "--max-kl", str(args.max_kl), "--min-top1",
                         str(args.min_top1)], log)
    lines = [ln for ln in out.strip().splitlines() if "positions |" in ln or ln.startswith(("PASS", "FAIL", "tokens"))]
    same_ids = any(ln.startswith("tokens:") and "identical to the reference" in ln for ln in lines)
    results["checks"].append({"check": "encoding with the model's special-token rule", "pass": same_ids,
                              "detail": next((ln for ln in lines if ln.startswith("tokens:")), "")})
    results["checks"].append({"check": "liyab-kl --kv f16 on the reference's ids", "pass": code == 0, "detail": lines})
    print(f"  the conformance text with the model's BOS rule: {'same ids' if same_ids else 'DIFFERENT ids'}")
    print(f"  forward pass against transformers: {'pass' if code == 0 else 'FAIL'}", flush=True)
    for ln in lines:
        print("    " + ln)
    results["row"] = code == 0
    ok = ok and same_ids and code == 0
    return ok, results


def has_tokenizer_json(model, revision):
    if os.path.isdir(model):
        return os.path.exists(os.path.join(model, "tokenizer.json"))
    from huggingface_hub import file_exists
    try:
        return file_exists(model, "tokenizer.json", revision=revision)
    except Exception:  # noqa: BLE001 - offline: assume the common case and let the check report
        return True


# ----------------------------------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------------------------------


def print_description(d, engine, arch):
    print("\nArchitecture facts (ArchSpec):")
    for name, _, _ in engine.fields:
        if name in d.fields:
            value, why = d.fields[name]
            shown = engine.format_value(name, value) if value != "" or engine.field_type(name) != "std::string_view" \
                else "(none)"
            print(f"  {name:<17} {shown:<28} {why}")
        else:
            print(f"  {name:<17} {'?':<28} not established")
    if d.parts:
        print("\nParts of this model (from the file, checked against the reference):")
        for p in d.parts:
            print("  " + p)
    if d.notes:
        print("\nNotes:")
        for n in d.notes:
            print("  " + n)
    if d.file_issues:
        print("\nThis GGUF disagrees with the reference (a problem of the file, not of the row):")
        for n in d.file_issues:
            print("  - " + n)
    if d.missing:
        print("\nNot implemented in Liyab (the model cannot run until these exist):")
        for n in dict.fromkeys(d.missing):
            print("  - " + n)
    if d.unknown:
        print("\nNot established (no row is proposed while any fact is unknown):")
        for n in dict.fromkeys(d.unknown):
            print("  - " + n)
    if d.row and d.ok:
        print("\nRegistry row (src/core/architecture.cpp, kArchitectures):\n" + d.row)
        if d.registry and d.registry.get("exists"):
            if d.registry["identical"]:
                print(f"The registry's row for {arch} is this row.")
            else:
                print(f"The registry already has a row for {arch} that differs: "
                      + ", ".join(f"{k} {v['registry']!r} there" for k, v in d.registry["differences"].items()))


def self_test(engine):
    """Every registry row parsed and printed back as architecture.cpp writes it."""
    failures = 0
    for name, (values, text) in engine.rows.items():
        printed = engine.format_row(values)
        same = " ".join(printed.split()) == " ".join((text + ",").split())
        failures += not same
        print(f"{name}: {'ok' if same else 'DIFFERS'}")
        if not same:
            print("  source:  " + " ".join(text.split()) + "\n  printed: " + " ".join(printed.split()))
    return failures == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Hugging Face model id or local directory (configuration, tokenizer, weights)")
    parser.add_argument("--gguf", help="GGUF of the model: a local file, or hf://<repo>/<file> (read remotely; "
                                       "--gate needs a local file)")
    parser.add_argument("--revision", help="model revision (branch, tag or commit)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="run modeling code shipped with the model (only for repositories you trust)")
    parser.add_argument("--gate", action="store_true", help="verify the row: build with it and run the conformance "
                                                            "checks (tokenizer ids, liyab-kl against transformers)")
    parser.add_argument("--source", default=REPO, help="Liyab source tree (default: this repository)")
    parser.add_argument("--scratch", default=os.path.join(REPO, "build", "describe_model"),
                        help="scratch copy and build for --gate (default: build/describe_model)")
    parser.add_argument("--max-kl", type=float, default=0.01, help="gate: mean KL limit (default 0.01)")
    parser.add_argument("--min-top1", type=float, default=97.0, help="gate: top-1 agreement floor (default 97)")
    parser.add_argument("--json", help="also write the description (and gate results) to this JSON file")
    parser.add_argument("--self-test", action="store_true", help="check the registry parser and printer, then exit")
    args = parser.parse_args()

    engine = Engine(args.source)
    if args.self_test:
        sys.exit(0 if self_test(engine) else 1)
    if not args.model or not args.gguf:
        parser.error("give --model and --gguf (or --self-test)")
    if args.gate and args.gguf.startswith("hf://"):
        parser.error("--gate needs the GGUF as a local file")

    print(f"Describing {args.model} with {args.gguf}", flush=True)
    try:
        d, row, g = describe(args.model, args.gguf, engine, args.revision, args.trust_remote_code)
    except (OSError, FormatError) as e:  # an unreadable or malformed GGUF
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    arch = g.arch
    print_description(d, engine, arch)
    out = {"model": args.model, "gguf": args.gguf, "architecture": arch, **d.to_json()}
    status = 0 if (d.ok and d.row) else 1
    if args.gate:
        if d.ok and d.row:
            passed, results = gate(args, d, d.row, arch, engine)
            out["gate"] = {"pass": passed, **results}
            if passed:
                print("\nACCEPTED: the row passes the conformance gate.")
            elif results.get("row"):
                print("\nREJECTED: the forward pass with the row matches the reference on its ids, but this GGUF "
                      "does not encode text as the model's tokenizer does (see above and the log).")
            else:
                print("\nREJECTED: the row does not pass the conformance gate (see the log).")
            status = 0 if passed else 1
        else:
            print("\nNo row to verify.")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=str)
    sys.exit(status)


if __name__ == "__main__":
    main()
