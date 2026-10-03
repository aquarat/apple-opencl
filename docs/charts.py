#!/usr/bin/env python3
"""Draw the README charts as plain SVG (no dependencies).

    python3 docs/charts.py      # writes docs/*-light.svg and docs/*-dark.svg

The numbers are the ones in STATE.md; edit them here and re-run. Light and
dark variants are separate files so the README's <picture> element can pick
the one matching the viewer's GitHub theme.
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Speed-up over stock Mesa 26.1.8 on the M1 mini: (label, detail, factor).
SPEEDUPS = [
    ("Read a GPU-written buffer back", "0.24 to 9.3 GB/s", 9.3 / 0.24),
    ("64 independent tiny kernels", "37 to 1.6 ms", 37 / 1.6),
    ("Read back a 371 MB image", "1.6 s to 76 ms", 1621 / 76),
    ("Chain of dependent tiny kernels", "38 to 4.2 µs per kernel", 38 / 4.2),
    ("Counted loop with a small body", "0.80 to 2.1 TFLOPS", 2093 / 795),
    ("Upload to a buffer the host reads (trade-off)", "24.6 to 11.3 GB/s", 11.3 / 24.6),
]

# darktable 5.6.1 export of a 24 MP raw: (label, seconds or None, note).
DARKTABLE = [
    ("CPU, 8 threads", 1.41, None),
    ("GPU, stock driver", None, "can't use the GPU: kernels fail to build"),
    ("GPU, this branch", 0.65, None),   # median of 0.611, 0.649, 0.652 s
]

THEMES = {
    # Surfaces are GitHub's; series colours from the validated reference
    # palette (blue/red), stepped per mode; grey is a deliberate neutral.
    "light": dict(text="#1f2328", muted="#59636e", grid="#d1d9e0",
                  faster="#2a78d6", slower="#e34948", neutral="#8c959f"),
    "dark": dict(text="#e6edf3", muted="#9198a1", grid="#3d444d",
                 faster="#3987e5", slower="#e66767", neutral="#6e7781"),
}

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', "
        "Helvetica, Arial, sans-serif")
W = 760
LABEL_W = 300          # left column: row labels
PLOT_X0, PLOT_X1 = LABEL_W + 12, W - 64
ROW_H, BAR_H = 34, 14


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar_path(x0, x1, y, h, r=4):
    """A bar anchored at x0 with its far end (x1, either side) rounded."""
    if abs(x1 - x0) < r:
        r = abs(x1 - x0) / 2
    if x1 >= x0:
        return (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
                f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z")
    return (f"M{x0:.1f},{y:.1f} H{x1 + r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 + r:.1f},{y + h:.1f} H{x0:.1f} Z")


def header(height, title, subtitle, c):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
        f'viewBox="0 0 {W} {height}" font-family="{FONT}" role="img" aria-label="{esc(title)}">',
        f'<title>{esc(title)}</title>',
        f'<text x="0" y="20" font-size="16" font-weight="600" fill="{c["text"]}">{esc(title)}</text>',
        f'<text x="0" y="40" font-size="12" fill="{c["muted"]}">{esc(subtitle)}</text>',
    ]


def speedups_svg(c):
    lo, hi = 0.25, 50.0
    def x(v):
        return PLOT_X0 + (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) * (PLOT_X1 - PLOT_X0)

    top = 64
    height = top + len(SPEEDUPS) * ROW_H + 30
    out = header(height, "Speed-up over stock Mesa (log scale)",
                 "Mac mini M1, 8-core GPU · microbenchmarks in bench/ · "
                 "vs Fedora Asahi's Mesa 26.1.8", c)
    plot_bottom = top + len(SPEEDUPS) * ROW_H
    for t in (0.5, 1, 2, 5, 10, 20, 50):
        xt = x(t)
        width = 1.5 if t == 1 else 1
        colour = c["muted"] if t == 1 else c["grid"]
        out.append(f'<line x1="{xt:.1f}" y1="{top - 6}" x2="{xt:.1f}" y2="{plot_bottom}" '
                   f'stroke="{colour}" stroke-width="{width}"/>')
        label = f"{t:g}×"
        out.append(f'<text x="{xt:.1f}" y="{plot_bottom + 16}" font-size="11" '
                   f'text-anchor="middle" fill="{c["muted"]}">{label}</text>')

    for i, (label, detail, f) in enumerate(SPEEDUPS):
        y = top + i * ROW_H
        by = y + (ROW_H - BAR_H) / 2
        out.append(f'<text x="0" y="{y + 15}" font-size="13" fill="{c["text"]}">{esc(label)}</text>')
        out.append(f'<text x="0" y="{y + 29}" font-size="11" fill="{c["muted"]}">{esc(detail)}</text>')
        x1 = x(f)
        colour = c["faster"] if f >= 1 else c["slower"]
        out.append(f'<path d="{bar_path(x(1), x1, by, BAR_H)}" fill="{colour}"/>')
        value = f"{f:.0f}×" if f >= 10 else f"{f:.1f}×" if f >= 1 else f"{f:.2f}×"
        if f >= 1:
            out.append(f'<text x="{x1 + 6:.1f}" y="{by + 11}" font-size="12" font-weight="600" '
                       f'fill="{c["text"]}">{value}</text>')
        else:
            out.append(f'<text x="{x1 - 6:.1f}" y="{by + 11}" font-size="12" font-weight="600" '
                       f'text-anchor="end" fill="{c["text"]}">{value}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def darktable_svg(c):
    hi = 1.5
    def x(v):
        return PLOT_X0 + v / hi * (PLOT_X1 - PLOT_X0)

    top = 64
    height = top + len(DARKTABLE) * ROW_H + 30
    out = header(height, "darktable export of a 24 MP raw (seconds, lower is better)",
                 "darktable 5.6.1 default pipeline · Mac mini M1 · "
                 "GPU output within 2/255 of the CPU export", c)
    plot_bottom = top + len(DARKTABLE) * ROW_H
    for t in (0, 0.5, 1.0, 1.5):
        xt = x(t)
        colour = c["muted"] if t == 0 else c["grid"]
        out.append(f'<line x1="{xt:.1f}" y1="{top - 6}" x2="{xt:.1f}" y2="{plot_bottom}" '
                   f'stroke="{colour}" stroke-width="{1.5 if t == 0 else 1}"/>')
        out.append(f'<text x="{xt:.1f}" y="{plot_bottom + 16}" font-size="11" '
                   f'text-anchor="middle" fill="{c["muted"]}">{t:g} s</text>')

    for i, (label, secs, note) in enumerate(DARKTABLE):
        y = top + i * ROW_H
        by = y + (ROW_H - BAR_H) / 2
        out.append(f'<text x="0" y="{y + 21}" font-size="13" fill="{c["text"]}">{esc(label)}</text>')
        if secs is None:
            out.append(f'<text x="{x(0) + 8:.1f}" y="{by + 11}" font-size="12" font-style="italic" '
                       f'fill="{c["muted"]}">{esc(note)}</text>')
            continue
        colour = c["neutral"] if label.startswith("CPU") else c["faster"]
        out.append(f'<path d="{bar_path(x(0), x(secs), by, BAR_H)}" fill="{colour}"/>')
        out.append(f'<text x="{x(secs) + 6:.1f}" y="{by + 11}" font-size="12" font-weight="600" '
                   f'fill="{c["text"]}">{secs:.2f} s</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for mode, c in THEMES.items():
        for name, fn in (("speedups", speedups_svg), ("darktable", darktable_svg)):
            path = os.path.join(HERE, f"{name}-{mode}.svg")
            with open(path, "w") as f:
                f.write(fn(c))
            print(path)
