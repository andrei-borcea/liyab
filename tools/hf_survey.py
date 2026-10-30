#!/usr/bin/env python3
"""What Hugging Face publishes, against what Liyab runs (standard library only, no token needed).

    tools/hf_survey.py fetch DIR [--repos 1000]    # download the listings into DIR (two JSON files)
    tools/hf_survey.py report DIR                  # print the survey from DIR

fetch: the most downloaded GGUF repositories (architecture, parameter count, pipeline, .gguf file names) and the
most downloaded text-generation repositories (config architecture, quantization_config, tensor dtypes), 100 per
request, following the Link header. The anonymous API allows 500 requests per 5 minutes: 1000 repositories take
ten requests per listing. Downloads are the all-time counts the API reports, so a survey is a snapshot: keep DIR to
compare two dates.

report:
  * GGUF repositories by architecture: share of the surveyed downloads, supported or not by Liyab (the registry
    kArchitectures in src/compat/architecture.cpp), and by pipeline (text, vision, audio...);
  * the quantization tags in .gguf file names (Q4_K_M, IQ2_XS, Q1_0...) against the tensor types Liyab recognizes
    (dtype_traits in src/core/types.cpp). A tag is matched by name, not by reading tensors: Q4_K_M is a mix whose
    tensors are Q4_K and Q6_K, and counts as recognized when its family is;
  * safetensors repositories: quantization methods and the dominant tensor dtype. Liyab reads GGUF only today.
"""
import argparse
import collections
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://huggingface.co/api/models"
GGUF_QUERY = "filter=gguf&sort=downloads&direction=-1&limit=100&" + "&".join(
    f"expand[]={f}" for f in ("gguf", "downloads", "siblings", "tags", "pipeline_tag"))
TEXT_QUERY = "pipeline_tag=text-generation&sort=downloads&direction=-1&limit=100&" + "&".join(
    f"expand[]={f}" for f in ("config", "downloads", "tags", "safetensors", "library_name"))
# Pipelines whose GGUF files are language models (or their embedding and ranking siblings).
TEXT_PIPELINES = {"text-generation", "image-text-to-text", "text2text-generation", "feature-extraction",
                  "sentence-similarity", "text-ranking", "fill-mask", "none"}
QUANT_TAG = re.compile(
    r"(?<![A-Za-z0-9])((?:UD-)?(?:IQ[1-4]_(?:XXS|XS|S|M|NL)|Q[1-8]_K(?:_(?:XL|XS|S|M|L))?|Q[1-8]_[01]|Q[1-8]_0_[0-9]_[0-9]"
    r"|TQ[12]_[01]|MXFP4(?:_MOE)?|NVFP4|BF16|F16|F32))(?![A-Za-z0-9])", re.I)


def get(url):
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "liyab-survey/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response), response.headers.get("Link", "")
        except urllib.error.HTTPError as error:
            wait = int(error.headers.get("Retry-After", 0) or 0) if error.code == 429 else 0
            last = error
            time.sleep(max(wait, 3 * (attempt + 1)))
        except OSError as error:
            last = error
            time.sleep(3 * (attempt + 1))
    raise SystemExit(f"{url}: {last}")


def listing(query, count):
    repos, url = [], f"{API}?{query}"
    while url and len(repos) < count:
        page, link = get(url)
        repos += page
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
        print(f"  {len(repos)} repositories", file=sys.stderr)
    return repos[:count]


def fetch(directory, count):
    directory.mkdir(parents=True, exist_ok=True)
    print("GGUF repositories", file=sys.stderr)
    gguf = []
    for r in listing(GGUF_QUERY, count):
        info = r.get("gguf") or {}
        gguf.append({"id": r["id"], "downloads": r.get("downloads") or 0, "arch": info.get("architecture"),
                     "params": info.get("total"), "pipeline": r.get("pipeline_tag"),
                     "files": [s["rfilename"] for s in r.get("siblings", []) if s["rfilename"].lower().endswith(".gguf")]})
    (directory / "gguf.json").write_text(json.dumps(gguf))
    print("text-generation repositories", file=sys.stderr)
    text = []
    for r in listing(TEXT_QUERY, count):
        config = r.get("config") or {}
        text.append({"id": r["id"], "downloads": r.get("downloads") or 0, "tags": r.get("tags", []),
                     "arch": (config.get("architectures") or [None])[0], "quant": config.get("quantization_config"),
                     "dtypes": (r.get("safetensors") or {}).get("parameters")})
    (directory / "text.json").write_text(json.dumps(text))


def liyab_architectures():
    source = (ROOT / "src/compat/architecture.cpp").read_text()
    table = re.search(r"kArchitectures\[\]\s*=\s*\{(.*?)\n\};", source, re.S).group(1)
    return set(re.findall(r'^\s*\{\s*(?:\.name\s*=\s*)?"([A-Za-z0-9_.\-]+)"', table, re.M))


def liyab_types():
    source = (ROOT / "src/core/types.cpp").read_text()
    return set(re.findall(r'return \{"([a-z0-9_]+)", \d+, \d+\};', source))


def recognized(tag, types):
    """Whether the tag names a mix of recognized types: the tag is a name, not a read of the tensors."""
    name = re.sub(r"(^ud-|_moe$)", "", tag.lower())  # Unsloth's dynamic prefix, the MXFP4 MoE mix
    if name in types:
        return True
    family = re.sub(r"_(xl|xs|s|m|l)$", "", name)  # q4_k_m -> q4_k, iq3_xs -> iq3 (a mix of IQ3 types)
    if family == "q8_k":  # the "8-bit K" mixes (Q8_K_XL...) are made of Q8_0 tensors: Q8_K is not a stored type
        family = "q8_0"
    return family in types or any(t.startswith(family + "_") for t in types if family.startswith("iq"))


def share(part, whole):
    return f"{100 * part / whole:5.1f}%" if whole else "    -"


def report(directory):
    gguf = json.loads((directory / "gguf.json").read_text())
    text = json.loads((directory / "text.json").read_text())
    ours, types = liyab_architectures(), liyab_types()
    total = sum(r["downloads"] for r in gguf)

    print(f"GGUF: {len(gguf)} repositories, {total:,} downloads\n")
    pipelines = collections.defaultdict(lambda: [0, 0])
    for r in gguf:
        row = pipelines[r["pipeline"] or "none"]
        row[0] += 1
        row[1] += r["downloads"]
    print("by pipeline (repositories, share of downloads)")
    for name, (n, d) in sorted(pipelines.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"  {name:30s} {n:4d} {share(d, total)}")

    language = [r for r in gguf if (r["pipeline"] or "none") in TEXT_PIPELINES]
    language_total = sum(r["downloads"] for r in language)
    missing = collections.defaultdict(list)
    for r in language:
        if r["arch"] not in ours:
            missing[r["arch"] or "?"].append(r)
    unsupported = sum(r["downloads"] for rs in missing.values() for r in rs)
    print(f"\nlanguage-model GGUFs: {share(language_total, total)} of all downloads; of those, architectures Liyab does not "
          f"run: {share(unsupported, language_total)}")
    print("not supported (repositories, share of all downloads, most downloaded repository)")
    for arch, rs in sorted(missing.items(), key=lambda kv: -sum(r["downloads"] for r in kv[1]))[:30]:
        print(f"  {arch:22s} {len(rs):3d} {share(sum(r['downloads'] for r in rs), total)}  {rs[0]['id']}")

    tags, tag_downloads = collections.Counter(), collections.Counter()
    for r in gguf:
        found = {m.upper() for f in r["files"] for m in QUANT_TAG.findall(f.rsplit("/", 1)[-1])}
        for tag in found:
            tags[tag] += 1
            tag_downloads[tag] += r["downloads"]
    print("\nquantization tags in file names (repositories offering it, share of downloads of those repositories)")
    for tag, n in tags.most_common(45):
        print(f"  {'ok ' if recognized(tag, types) else 'NO '} {tag:14s} {n:4d} {share(tag_downloads[tag], total)}")
    unknown = [t for t in tags if not recognized(t, types)]
    print(f"tags Liyab does not recognize: {', '.join(sorted(unknown)) or 'none'}")

    print(f"\nsafetensors (text-generation): {len(text)} repositories, {sum(r['downloads'] for r in text):,} downloads")
    text_total = sum(r["downloads"] for r in text)
    methods, dtypes = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
    for r in text:
        quant = r["quant"] if isinstance(r["quant"], dict) else {}
        method = quant.get("quant_method") or "none (plain weights)"
        methods[method][0] += 1
        methods[method][1] += r["downloads"]
        if r["dtypes"]:
            dominant = max(r["dtypes"], key=lambda k: r["dtypes"][k])
            dtypes[dominant][0] += 1
            dtypes[dominant][1] += r["downloads"]
    print("quantization_config.quant_method (repositories, share of downloads)")
    for name, (n, d) in sorted(methods.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"  {name:26s} {n:4d} {share(d, text_total)}")
    print("dominant tensor dtype (repositories, share of downloads)")
    for name, (n, d) in sorted(dtypes.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"  {name:26s} {n:4d} {share(d, text_total)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download the listings")
    f.add_argument("directory", type=Path)
    f.add_argument("--repos", type=int, default=1000, help="repositories per listing (default 1000)")
    r = sub.add_parser("report", help="print the survey")
    r.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.directory, args.repos)
    else:
        report(args.directory)


if __name__ == "__main__":
    main()
