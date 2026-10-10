#!/usr/bin/env python3
"""Draws Liyab's animated brand images for the README (docs/brand/*.svg).

The flame is the app's living flame (app/lib/ui/living_flame.dart) translated
to SVG/SMIL: the same paths, colour ramps, sway rates and amplitudes per
state, and sparks with the same lifetime and speeds. The palette and type come
from app/lib/ui/theme.dart and app/assets/fonts. SMIL animations play inside
<img>, so the images animate on GitHub without scripts.

The fonts are embedded as base64 WOFF subsets of the characters each image
uses (Bricolage Grotesque for display, Figtree for text; both SIL OFL, see
app/assets/fonts), so the images render in the brand's type everywhere.

Usage:  pip install fonttools && python3 tools/gen_brand_svgs.py
Writes: docs/brand/liyab-banner.svg, liyab-flame.svg, liyab-states.svg,
        liyab-heat.svg, liyab-palette.svg, liyab-social.svg (the repository's social
        preview; GitHub needs it as PNG: liyab-social.png, a 1280x640 screenshot of it)
"""
import base64
import io
import math
import random
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "app" / "assets" / "fonts"
OUT = ROOT / "docs" / "brand"

# app/lib/ui/theme.dart (Palette)
KILN, KILN_RAISED, HAIRLINE = "#1B1420", "#251C2B", "#3A2E41"
ASH, SMOKE = "#EDE6EE", "#A99BAD"
FLARE, EMBER, GOLD, CORE = "#FF3D6E", "#FF7A3D", "#FFC24D", "#9ED8FF"
DAY_GROUND, DAY_RAISED, DAY_HAIRLINE, DAY_TEXT, DAY_SMOKE = "#F7F2F5", "#FFFFFF", "#E3D8E2", "#2A1F2E", "#6E5F73"

# living_flame.dart: colour ramps (cool, warm, hot) per gradient stop.
RAMPS = {
    "left_base": ("#6C7BFF", "#FF3D6E", "#E3263B"),
    "left_tip": ("#9C8CFF", "#FF8A3D", "#FF5A2A"),
    "right_base": ("#8FA6FF", "#FF7A3D", "#FF4A2E"),
    "right_tip": ("#B9D4FF", "#FFC24D", "#FF9A3D"),
    "core_edge": ("#BFE6FF", "#9ED8FF", "#FFE0B0"),
}

LEFT = "M246 48 C 166 144 132 220 132 302 C 132 386 192 446 262 458 C 212 428 186 378 190 322 C 196 252 252 196 246 48 Z"
RIGHT = "M304 144 C 364 210 386 262 386 322 C 386 396 332 450 262 458 C 308 430 332 390 328 344 C 324 290 296 236 304 144 Z"

# living_flame.dart: per state, sway rate (rad/s), amplitude (degrees), spark emission per frame and sparks per emission.
STATES = {
    "resting": dict(rate=0.7, amp=1.2, emit=0.03, burst=1),
    "listening": dict(rate=1.4, amp=2.2, emit=0.12, burst=1),
    "thinking": dict(rate=2.2, amp=2.6, emit=1.6, burst=3),
    "answering": dict(rate=3.4, amp=3.2, emit=0.5, burst=1),
}


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def lerp_hex(a, b, t):
    ra, rb = hex_rgb(a), hex_rgb(b)
    return "#%02X%02X%02X" % tuple(round(x + (y - x) * t) for x, y in zip(ra, rb))


def ramp(name, heat):
    cool, warm, hot = RAMPS[name]
    return lerp_hex(cool, warm, heat / 0.5) if heat < 0.5 else lerp_hex(warm, hot, (heat - 0.5) / 0.5)


def font_face(file, family, axes, text):
    """A @font-face rule with `file` instanced at `axes`, subset to `text`, as base64 WOFF."""
    font = instancer.instantiateVariableFont(TTFont(FONTS / file), axes)
    options = subset.Options()
    options.flavor = "woff"
    options.layout_features = ["kern", "liga"]
    sub = subset.Subsetter(options)
    sub.populate(text=text + " ")
    sub.subset(font)
    buf = io.BytesIO()
    font.flavor = "woff"
    font.recalcTimestamp = False  # the same input gives the same bytes (no "modified" timestamp)
    font["head"].modified = font["head"].created
    font.save(buf)
    data = base64.b64encode(buf.getvalue()).decode()
    return f"@font-face{{font-family:'{family}';src:url(data:font/woff;base64,{data}) format('woff');}}"


def style(display_text="", body_text="", strong_text=""):
    rules = []
    if display_text:
        rules.append(font_face("BricolageGrotesque.ttf", "LiyabDisplay", {"wght": 700, "opsz": 96, "wdth": 100}, display_text))
    if body_text:
        rules.append(font_face("Figtree.ttf", "LiyabText", {"wght": 500}, body_text))
    if strong_text:
        rules.append(font_face("Figtree.ttf", "LiyabStrong", {"wght": 700}, strong_text))
    rules.append(".d{font-family:LiyabDisplay,'Bricolage Grotesque',system-ui,sans-serif}"
                 ".t{font-family:LiyabText,Figtree,system-ui,sans-serif}"
                 ".s{font-family:LiyabStrong,Figtree,system-ui,sans-serif}")
    return "<style>" + "".join(rules) + "</style>"


def samples(fn, period, n):
    """values / keyTimes for a SMIL animation of fn(t) over one period."""
    vals = [fn(period * i / n) for i in range(n + 1)]
    return vals, ";".join(f"{i / n:.4f}" for i in range(n + 1))


def flame(pid, state="resting", heat=0.5, heat_cycle=None, seed=1):
    """The living flame in its 512-unit viewport, as an SVG group. `heat_cycle`: seconds for cool→hot→cool."""
    p = STATES[state]
    rate, amp = p["rate"], p["amp"]
    period = 2 * math.pi / rate
    lean = 4.6 if state == "listening" else 0.0  # 3 + voice * 4 at a voice level of 0.4

    def flick(t):
        return math.sin(t * 17) * 1.4 if state == "answering" else 0.0

    n = 60 if state == "answering" else 32
    left_vals, keys = samples(lambda t: math.sin(t * rate) * amp - lean + flick(t), period, n)
    right_vals, _ = samples(lambda t: -math.sin(t * rate) * amp * 0.8 + lean - flick(t), period, n)
    core_amp = 1.5 if state == "resting" else 2.5
    voice = (lambda t: 8 * (0.5 + 0.5 * math.sin(t * rate * 1.5))) if state == "listening" else (lambda t: 0.0)
    core_vals, _ = samples(lambda t: 36 + voice(t) + math.sin(t * rate * 2) * core_amp, period, n)

    def stop_color(name, offset):
        if heat_cycle:
            cool, warm, hot = RAMPS[name]
            vals = ";".join([cool, warm, hot, warm, cool])
            return (f'<stop offset="{offset}" stop-color="{warm}">'
                    f'<animate attributeName="stop-color" values="{vals}" dur="{heat_cycle}s" repeatCount="indefinite"/></stop>')
        return f'<stop offset="{offset}" stop-color="{ramp(name, heat)}"/>'

    def rotate(values):
        v = ";".join(f"{x:.2f} 262 458" for x in values)
        return (f'<animateTransform attributeName="transform" type="rotate" values="{v}" keyTimes="{keys}" '
                f'dur="{period:.3f}s" repeatCount="indefinite"/>')

    halo_alpha = 0.30 + heat * 0.25
    halo_r = 110 + 36 * 1.2 + heat * 30
    out = [f'<defs>'
           f'<linearGradient id="{pid}L" gradientUnits="userSpaceOnUse" x1="132" y1="458" x2="197" y2="48">'
           f'{stop_color("left_base", 0)}{stop_color("left_tip", 1)}</linearGradient>'
           f'<linearGradient id="{pid}R" gradientUnits="userSpaceOnUse" x1="262" y1="458" x2="324" y2="144">'
           f'{stop_color("right_base", 0)}{stop_color("right_tip", 1)}</linearGradient>'
           f'<radialGradient id="{pid}C" cx="0.4" cy="0.35" r="0.75"><stop offset="0" stop-color="#FFFFFF"/>'
           f'{stop_color("core_edge", 1)}</radialGradient>'
           f'<radialGradient id="{pid}H"><stop offset="0" stop-color="{ramp("left_tip", heat)}" stop-opacity="{halo_alpha:.2f}">'
           + (f'<animate attributeName="stop-color" values="{";".join(RAMPS["left_tip"] + RAMPS["left_tip"][1::-1])}" '
              f'dur="{heat_cycle}s" repeatCount="indefinite"/>' if heat_cycle else "")
           + f'</stop><stop offset="1" stop-color="{ramp("left_tip", heat)}" stop-opacity="0"/></radialGradient>'
           f'</defs>']
    halo_vals = ";".join(f"{halo_r + (r - 36) * 1.2:.1f}" for r in core_vals)
    out.append(f'<circle cx="262" cy="356" r="{halo_r:.1f}" fill="url(#{pid}H)">'
               f'<animate attributeName="r" values="{halo_vals}" keyTimes="{keys}" dur="{period:.3f}s" repeatCount="indefinite"/></circle>')
    out.append(f'<path d="{LEFT}" fill="url(#{pid}L)">{rotate(left_vals)}</path>')
    out.append(f'<path d="{RIGHT}" fill="url(#{pid}R)">{rotate(right_vals)}</path>')
    out.append(f'<circle cx="262" cy="356" r="36" fill="url(#{pid}C)">'
               f'<animate attributeName="r" values="{";".join(f"{r:.2f}" for r in core_vals)}" keyTimes="{keys}" '
               f'dur="{period:.3f}s" repeatCount="indefinite"/></circle>')

    # Sparks: a life of 1/0.012 frames at ~30 frames a second, rising 1.6..4.8 units a frame with a sideways wobble.
    rng = random.Random(seed)
    life_s = (1 / 0.012) / 30
    per_second = p["emit"] * p["burst"] * 30
    count = max(1, min(14, round(per_second * life_s / 6)))  # a readable share of what the app draws
    for i in range(count):
        x0 = 262 + (rng.random() - 0.5) * 30
        vy = 1.6 + rng.random() * 3.2
        vx = (rng.random() - 0.5) * 1.6
        frames = int(1 / 0.012)
        xs, ys = [], []
        x, y = x0, 356.0
        for f in range(0, frames + 1, frames // 8):
            xs.append(x0 + vx * f + math.sin((356 - vy * f + f / 30 * 40) / 18) * 0.6 * f / 4)
            ys.append(356 - vy * f)
        cool = heat_cycle is None and heat < 0.5
        color = "#BFD8FF" if cool else "#FFC85A"
        begin = -life_s * i / count - rng.random() * 0.3
        dur = f"{life_s * (1.5 if state == 'resting' else 1):.2f}s"
        out.append(f'<circle r="6" fill="{color}" opacity="0">'
                   f'<animate attributeName="cx" values="{";".join(f"{v:.1f}" for v in xs)}" dur="{dur}" begin="{begin:.2f}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="cy" values="{";".join(f"{v:.1f}" for v in ys)}" dur="{dur}" begin="{begin:.2f}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="r" values="6;2.4" dur="{dur}" begin="{begin:.2f}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="opacity" values="0;1;0" keyTimes="0;0.08;1" dur="{dur}" begin="{begin:.2f}s" repeatCount="indefinite"/>'
                   f'</circle>')
    return "".join(out)


def svg(width, height, body, title, defs=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="{title}"><title>{title}</title>{defs}{body}</svg>\n')


def placed(x, y, size, inner):
    return f'<g transform="translate({x} {y}) scale({size / 512:.4f})">{inner}</g>'


def banner():
    title, tagline = "Liyab", "Large language models on the device in your pocket."
    points = "fast   ·   cool to the touch   ·   private by construction"
    body = (f'<rect width="1280" height="420" rx="36" fill="{KILN}"/>'
            f'<rect x="0.5" y="0.5" width="1279" height="419" rx="36" fill="none" stroke="{HAIRLINE}"/>'
            + placed(70, 30, 360, flame("b", "thinking", heat=0.45, seed=3))
            + f'<text x="460" y="215" class="d" font-size="150" fill="{ASH}" letter-spacing="-4">{title}</text>'
            f'<text x="466" y="275" class="t" font-size="30" fill="{ASH}">{tagline}</text>'
            f'<text x="466" y="330" class="t" font-size="22" fill="{SMOKE}" letter-spacing="1">{points}</text>'
            f'<rect x="466" y="356" width="140" height="5" rx="2.5" fill="url(#bar)"/>')
    defs = (style(title, tagline + points) +
            f'<defs><linearGradient id="bar"><stop offset="0" stop-color="{FLARE}"/><stop offset="0.5" stop-color="{EMBER}"/>'
            f'<stop offset="1" stop-color="{GOLD}"/></linearGradient></defs>')
    return svg(1280, 420, body, "Liyab: large language models on the device in your pocket", defs)


def flame_only():
    return svg(512, 512, flame("f", "resting", heat=0.45, seed=7), "The Liyab flame")


def states():
    labels = [("resting", "Resting", "breathes slowly"), ("listening", "Listening", "leans toward your voice"),
              ("thinking", "Thinking", "sparks rise as it reasons"), ("answering", "Answering", "flickers as words stream")]
    body = [f'<rect width="1280" height="400" rx="36" fill="{KILN}"/>']
    text = ""
    for i, (state, name, line) in enumerate(labels):
        cx = 160 + i * 320
        body.append(placed(cx - 130, 20, 260, flame(f"s{i}", state, heat=0.45, seed=11 + i)))
        body.append(f'<text x="{cx}" y="320" text-anchor="middle" class="d" font-size="34" fill="{ASH}">{name}</text>')
        body.append(f'<text x="{cx}" y="356" text-anchor="middle" class="t" font-size="20" fill="{SMOKE}">{line}</text>')
        text += name + line
    names = "".join(n for _, n, _ in labels)
    return svg(1280, 400, "".join(body), "The living flame's four states: resting, listening, thinking, answering",
               style(names, text))


def heat():
    cycle = 12
    body = [f'<rect width="1280" height="360" rx="36" fill="{KILN}"/>',
            placed(40, 20, 320, flame("h", "answering", heat_cycle=cycle, seed=21))]
    # The ramp of the left blade's tip, cool → warm → hot, with a marker in step with the flame's colour.
    stops = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in zip((0, 0.5, 1), RAMPS["left_tip"]))
    body.append(f'<defs><linearGradient id="ramp">{stops}</linearGradient></defs>')
    body.append(f'<text x="420" y="96" class="d" font-size="40" fill="{ASH}">Its colour follows the phone\'s heat</text>')
    body.append(f'<text x="420" y="138" class="t" font-size="22" fill="{SMOKE}">Driven by the OS\'s 10-second thermal-headroom forecast:</text>')
    body.append(f'<text x="420" y="168" class="t" font-size="22" fill="{SMOKE}">cool while there is room, hot as throttling gets near.</text>')
    body.append(f'<rect x="420" y="208" width="800" height="22" rx="11" fill="url(#ramp)"/>')
    body.append(f'<circle cy="219" r="17" fill="{KILN}" stroke="{ASH}" stroke-width="4">'
                f'<animate attributeName="cx" values="428;1212;428" keyTimes="0;0.5;1" dur="{cycle}s" repeatCount="indefinite" '
                f'calcMode="spline" keySplines="0.45 0 0.55 1;0.45 0 0.55 1"/></circle>')
    for x, anchor, label in ((420, "start", "cool · plenty of headroom"), (820, "middle", "warm"),
                             (1220, "end", "hot · near throttling")):
        body.append(f'<text x="{x}" y="270" text-anchor="{anchor}" class="t" font-size="20" fill="{SMOKE}">{label}</text>')
    text = ("Driven by the OS's 10-second thermal-headroom forecast:cool while there is room, hot as throttling gets near."
            "cool · plenty of headroomwarmhot · near throttling")
    return svg(1280, 360, "".join(body), "The flame's colour follows the phone's thermal headroom",
               style("Its colour follows the phone's heat", text))


def palette():
    night = [("Kiln", KILN, "night ground"), ("Kiln raised", KILN_RAISED, "sheets, input"),
             ("Hairline", HAIRLINE, "dividers"), ("Ash", ASH, "text on night"), ("Smoke", SMOKE, "secondary text")]
    fire = [("Flare", FLARE, "secondary"), ("Ember", EMBER, "primary action"), ("Gold", GOLD, "tertiary"),
            ("Core", CORE, "the light inside")]
    day = [("Day ground", DAY_GROUND, ""), ("Day raised", DAY_RAISED, ""), ("Day hairline", DAY_HAIRLINE, ""),
           ("Day text", DAY_TEXT, ""), ("Day smoke", DAY_SMOKE, "")]
    body = [f'<rect width="1280" height="760" rx="36" fill="{KILN}"/>']
    text, strong = "", ""

    def row(y, items, label, w):
        nonlocal text, strong
        body.append(f'<text x="60" y="{y - 22}" class="s" font-size="18" fill="{SMOKE}" letter-spacing="3">{label}</text>')
        strong += label
        gap = 20
        for i, (name, color, role) in enumerate(items):
            x = 60 + i * (w + gap)
            ink = ASH if hex_rgb(color)[1] < 120 or color in (KILN, KILN_RAISED, HAIRLINE, DAY_TEXT, DAY_SMOKE) else KILN
            body.append(f'<rect x="{x}" y="{y}" width="{w}" height="120" rx="20" fill="{color}" stroke="{HAIRLINE}"/>')
            body.append(f'<text x="{x + 18}" y="{y + 40}" class="s" font-size="20" fill="{ink}">{name}</text>')
            body.append(f'<text x="{x + 18}" y="{y + 68}" class="t" font-size="17" fill="{ink}" opacity="0.85">{color}</text>')
            if role:
                body.append(f'<text x="{x + 18}" y="{y + 100}" class="t" font-size="15" fill="{ink}" opacity="0.7">{role}</text>')
            text += color + role
            strong += name

    row(80, night, "NIGHT", 216)
    row(270, fire, "FLAME", 276)
    row(460, day, "DAY", 216)
    body.append(f'<text x="60" y="672" class="d" font-size="44" fill="{ASH}">Bricolage Grotesque</text>')
    body.append(f'<text x="560" y="672" class="t" font-size="20" fill="{SMOKE}">display · headings · the wordmark</text>')
    body.append(f'<text x="60" y="718" class="t" font-size="30" fill="{ASH}">Figtree for everything you read</text>')
    body.append(f'<text x="560" y="718" class="t" font-size="20" fill="{SMOKE}">text · labels · tabular figures</text>')
    text += "display · headings · the wordmarkFigtree for everything you readtext · labels · tabular figures"
    return svg(1280, 760, "".join(body), "Liyab colour palette and typography",
               style("Bricolage Grotesque", text, strong))


def social():
    """The repository's social preview (1280x640, GitHub's size), rasterized to PNG for upload."""
    title, tagline = "Liyab", "Large open language models on the device in your pocket"
    facts = ["A 22 GB mixture-of-experts model, streamed from flash, on a phone",
             "Paced by the phone's thermal forecast: cool to the touch",
             "Private by construction: nothing leaves the device"]
    body = [f'<rect width="1280" height="640" fill="{KILN}"/>',
            f'<rect x="0" y="628" width="1280" height="12" fill="url(#bar)"/>',
            placed(40, 110, 420, flame("p", "thinking", heat=0.45, seed=5)),
            f'<text x="480" y="250" class="d" font-size="150" fill="{ASH}" letter-spacing="-4">{title}</text>',
            f'<text x="486" y="310" class="t" font-size="27" fill="{ASH}">{tagline}</text>']
    for i, fact in enumerate(facts):
        y = 390 + i * 52
        body.append(f'<circle cx="496" cy="{y - 9}" r="6" fill="{(FLARE, EMBER, GOLD)[i]}"/>')
        body.append(f'<text x="516" y="{y}" class="t" font-size="24" fill="{SMOKE}">{fact}</text>')
    body.append(f'<text x="486" y="580" class="s" font-size="22" fill="{EMBER}">github.com/andrei-borcea/liyab</text>')
    defs = (style(title, tagline + "".join(facts), "github.com/andrei-borcea/liyab") +
            f'<defs><linearGradient id="bar"><stop offset="0" stop-color="{FLARE}"/><stop offset="0.5" stop-color="{EMBER}"/>'
            f'<stop offset="1" stop-color="{GOLD}"/></linearGradient></defs>')
    return svg(1280, 640, "".join(body), "Liyab: large open language models on the device in your pocket", defs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (("liyab-banner.svg", banner), ("liyab-flame.svg", flame_only), ("liyab-states.svg", states),
                       ("liyab-heat.svg", heat), ("liyab-palette.svg", palette), ("liyab-social.svg", social)):
        data = make()
        (OUT / name).write_text(data)
        print(f"{name}: {len(data) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
