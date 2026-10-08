#!/usr/bin/env python3
"""Renders the README benchmark charts (SVG) from docs/benchmarks/results.json.

    python3 tools/gen_bench_charts.py            # writes docs/benchmarks/*.svg

No dependencies. Each SVG carries light and dark colors (prefers-color-scheme),
direct value labels on every bar (two light-mode series colors are below 3:1
against the surface, so values never rely on the bar color alone) and a legend
whenever a chart has more than one series. The README keeps the same numbers as
tables. Palette: a CVD-validated categorical order (blue, orange, aqua,
yellow), checked in both modes.
"""
import json
import math
import os
from html import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "docs", "benchmarks", "results.json")
OUT = os.path.join(ROOT, "docs", "benchmarks")

LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
         "axis": "#c3c2b7", "s": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]}
DARK = {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "axis": "#383835", "s": ["#3987e5", "#d95926", "#199e70", "#c98500"]}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def style():
    def block(c):
        rules = [f".bg{{fill:{c['surface']}}}", f".t1{{fill:{c['ink']}}}", f".t2{{fill:{c['ink2']}}}",
                 f".tm{{fill:{c['muted']}}}", f".grid{{stroke:{c['grid']}}}", f".axis{{stroke:{c['axis']}}}"]
        for i, col in enumerate(c["s"]):
            rules.append(f".f{i}{{fill:{col}}} .k{i}{{stroke:{col}}} .gap{{stroke:{c['surface']}}}")
        return "".join(rules)
    return (f"<style>text{{font-family:{FONT};}} .num{{font-variant-numeric:tabular-nums}} {block(LIGHT)}"
            f" @media (prefers-color-scheme: dark){{ {block(DARK)} }}</style>")


def svg(width, height, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"'
            f' role="img" aria-label="{escape(title)}"><title>{escape(title)}</title>{style()}'
            f'<rect class="bg" width="{width}" height="{height}" rx="8"/>{body}</svg>\n')


def text(x, y, s, cls="t2", size=12, anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" text-anchor="{anchor}"'
            f' font-weight="{weight}">{escape(str(s))}</text>')


def hbar(x, y, w, h, cls):
    """Horizontal bar anchored at x, 4px rounded corners on the data end only."""
    if w <= 0:
        return ""
    r = min(4.0, w, h / 2)
    return (f'<path class="{cls}" d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}'
            f' V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z"/>')


def nice_max(v):
    exp = 10 ** math.floor(math.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * exp >= v:
            return m * exp
    return 10 * exp


def fmt(v):
    return f"{v:.2f}" if v < 1 else (f"{v:.1f}" if v < 100 else f"{v:.0f}")


def tick(v, vmax):
    return f"{v:.0f}" if vmax >= 5 else f"{v:.1f}"


DEVICE = ""  # footnote on every chart, set from the data file


def footnote(width, height):
    return text(16, height - 8, DEVICE, "tm", 10)


def legend(items, x, y):
    out, cx = [], x
    for i, label in enumerate(items):
        out.append(f'<rect class="f{i}" x="{cx}" y="{y - 9}" width="10" height="10" rx="2"/>')
        out.append(text(cx + 15, y, label, "t2", 12))
        cx += 15 + 7.2 * len(label) + 22
    return "".join(out)


def grouped_hbar_chart(title, subtitle, categories, series, values, unit, path, width=760):
    """Horizontal grouped bars: one group per category, one bar per series."""
    # Left margin fits the longest category label (~6.8 px per character at 12 px).
    left = max(120, 16 + int(6.8 * max(len(c) for c in categories)) + 14)
    right, top = 70, 70 + (22 if len(series) > 1 else 0)
    bar_h, gap, group_gap = 14, 2, 16
    n = len(series)
    group_h = n * bar_h + (n - 1) * gap
    height = top + len(categories) * (group_h + group_gap) + 58
    vmax = nice_max(max(v for row in values for v in row if v is not None))
    plot_w = width - left - right
    sx = lambda v: plot_w * v / vmax
    body = [text(16, 26, title, "t1", 15, weight=600), text(16, 46, subtitle, "t2", 12)]
    if n > 1:
        body.append(legend(series, 16, 74))
    ticks = 5
    for i in range(ticks + 1):  # recessive grid, value axis at the bottom
        v = vmax * i / ticks
        x = left + sx(v)
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{height - 52}" stroke-width="1"/>')
        body.append(text(x, height - 36, tick(v, vmax), "tm num", 11, "middle"))
    body.append(text(left + plot_w, height - 22, unit, "tm", 11, "end"))
    body.append(footnote(width, height))
    y = top
    for c, row in zip(categories, values):
        body.append(text(left - 10, y + group_h / 2 + 4, c, "t1", 12, "end"))
        for i, v in enumerate(row):
            by = y + i * (bar_h + gap)
            if v is None:
                body.append(text(left + 4, by + bar_h - 3, "n/a", "tm", 11))
                continue
            body.append(hbar(left, by, sx(v), bar_h, f"f{i}"))
            body.append(text(left + sx(v) + 6, by + bar_h - 3, fmt(v), "t1 num", 11))
        y += group_h + group_gap
    body.append(f'<line class="axis" x1="{left}" y1="{top - 6}" x2="{left}" y2="{height - 52}" stroke-width="1"/>')
    with open(path, "w") as f:
        f.write(svg(width, height, "".join(body), title))


def line_chart(title, subtitle, xs, xlabels, series, unit, xunit, path, width=760, height=400):
    left, right, top, bottom = 56, 110, 112, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    vmax = nice_max(max(v for vals in series.values() for v in vals))
    sx = lambda i: left + plot_w * i / (len(xs) - 1)
    sy = lambda v: top + plot_h * (1 - v / vmax)
    body = [text(16, 26, title, "t1", 15, weight=600), text(16, 46, subtitle, "t2", 12),
            legend(list(series), 16, 74)]
    for i in range(6):
        v = vmax * i / 5
        body.append(f'<line class="grid" x1="{left}" y1="{sy(v):.1f}" x2="{left + plot_w}" y2="{sy(v):.1f}" stroke-width="1"/>')
        body.append(text(left - 8, sy(v) + 4, tick(v, vmax), "tm num", 11, "end"))
    for i, lab in enumerate(xlabels):
        body.append(text(sx(i), top + plot_h + 18, lab, "tm num", 11, "middle"))
    body.append(text(left + plot_w / 2, top + plot_h + 38, xunit, "tm", 11, "middle"))
    body.append(text(left - 8, top - 14, unit, "tm", 11, "end"))
    body.append(footnote(width, height))
    body.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke-width="1"/>')
    # Direct labels at the right end, nudged apart so they never collide.
    ends = sorted(((sy(vals[-1]), i, name) for i, (name, vals) in enumerate(series.items())), key=lambda t: t[0])
    placed = []
    for yv, i, name in ends:
        yl = max(yv, placed[-1] + 14) if placed else yv
        placed.append(yl)
        body.append(text(left + plot_w + 10, yl + 4, name, "t2", 11))
    for i, (name, vals) in enumerate(series.items()):
        pts = " ".join(f"{sx(j):.1f},{sy(v):.1f}" for j, v in enumerate(vals))
        body.append(f'<polyline class="k{i}" fill="none" stroke-width="2" stroke-linejoin="round" points="{pts}"/>')
        for j, v in enumerate(vals):  # markers with a 2px surface ring where lines overlap
            body.append(f'<circle class="f{i} gap" cx="{sx(j):.1f}" cy="{sy(v):.1f}" r="4" stroke-width="2"/>')
    with open(path, "w") as f:
        f.write(svg(width, height, "".join(body), title))


def main():
    global DEVICE
    with open(DATA) as f:
        d = json.load(f)
    DEVICE = "Measured on " + d["device"] + "."

    r = d["dense_27b"]
    grouped_hbar_chart(r["title"] + ": decode speed", "Higher is better.",
                       [x["label"] for x in r["runs"]], ["decode"], [[x["decode_tps"]] for x in r["runs"]],
                       "tokens / s", os.path.join(OUT, "dense_27b_decode.svg"))
    grouped_hbar_chart(r["title"] + ": prompt time (18 tokens)", "Lower is better.",
                       [x["label"] for x in r["runs"]], ["prompt"], [[x["prompt_s"]] for x in r["runs"]],
                       "seconds", os.path.join(OUT, "dense_27b_prompt.svg"))

    moe = d["moe_35b"]
    grouped_hbar_chart(moe["title"] + ": decode speed", "Higher is better. Each step adds to the previous one.",
                       [x["label"] for x in moe["runs"]], ["decode"], [[x["decode_tps"]] for x in moe["runs"]],
                       "tokens / s", os.path.join(OUT, "moe_35b_decode.svg"))
    ph = moe["phases_ms_per_token"]
    grouped_hbar_chart("Qwen3.6-35B-A3B: decode time per token by phase", "Lower is better.", list(ph["phases"]),
                       ph["series"], list(ph["phases"].values()), "ms / token",
                       os.path.join(OUT, "moe_35b_phases.svg"))

    fl = d["flash"]
    labels = [f"{k} KiB" if k < 1024 else f"{k // 1024} MiB" for k in fl["chunk_kib"]]
    line_chart(fl["title"], "Requests of 256 KiB or more, 2-4 in flight, saturate the device; small requests do not.",
               fl["chunk_kib"], labels, fl["gbps_by_threads"], "GB/s", "request size",
               os.path.join(OUT, "flash_reads.svg"))

    m = d["matvec"]
    grouped_hbar_chart(m["title"], "4 CPU threads vs the Adreno 830 GPU; best of several runs.",
                       list(m["formats"]), m["series"], list(m["formats"].values()), "GB/s",
                       os.path.join(OUT, "matvec_formats.svg"))

    ram = d["in_ram"]
    grouped_hbar_chart(ram["title"], "Higher is better; n/a: not measured.", list(ram["models"]), ram["series"],
                       list(ram["models"].values()), "tokens / s", os.path.join(OUT, "in_ram_decode.svg"))
    print("wrote", ", ".join(sorted(p for p in os.listdir(OUT) if p.endswith(".svg"))))


if __name__ == "__main__":
    main()
