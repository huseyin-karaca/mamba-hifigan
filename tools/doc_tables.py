"""Write the site's tables and loss-curve figure from results/.

Everything the documentation states as a number comes from here, so the page and the CSVs cannot drift apart.
Run it after changing results/experiments.csv or results/loss_curves.csv:

    python tools/doc_tables.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNIP = ROOT / "docs" / "snippets"

FAMILY_TITLE = {
    "baseline": "Baseline",
    "gated": "Adaptive gated fusion (proposed)",
    "parallel": "Parallel branches, fixed fusion",
    "ablation": "Mamba only",
    "phase": "Phase coherence failures",
    "gradient": "Gradient instability failures",
    "bottleneck": "Information bottleneck failures",
    "partial": "Mamba at some resolutions only",
}
FAMILY_ORDER = list(FAMILY_TITLE)


def load():
    with open(ROOT / "results" / "experiments.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["fm"] = float(r["g_feat_match_loss"])
        r["l1"] = float(r["g_l1_spec_loss"])
    return rows


def fmt(v):
    return f"{v:.4f}"


def cell(v, best):
    cls = "num best" if abs(v - best) < 1e-9 else "num"
    return f'<td class="{cls}">{fmt(v)}</td>'


def table_main(rows):
    """The five configurations the report leads with."""
    want = [
        "Baseline HiFi-GAN",
        "Mamba only, no MRF",
        "Hybrid 2 summation, Mamba-64",
        "Hybrid 1 gated, Mamba-64",
        "Hybrid 1 gated, Mamba-128",
    ]
    sel = [next(r for r in rows if r["label"] == lab) for lab in want]
    bfm, bl1 = min(r["fm"] for r in sel), min(r["l1"] for r in sel)
    base = next(r for r in sel if r["family"] == "baseline")

    out = ['<div class="mh-table" markdown="0">', "<table>", "<thead><tr>",
           "<th>Configuration</th><th class=\"num\">Feature matching ↓</th>",
           "<th class=\"num\">L1 spectrogram ↓</th><th class=\"num\">FM vs baseline</th>",
           "</tr></thead>", "<tbody>"]
    for r in sel:
        hi = ' class="mh-row"' if r["family"] == "gated" else ""
        if r is base:
            delta = "—"
        else:
            d = (r["fm"] - base["fm"]) / base["fm"] * 100
            delta = f"{d:+.1f}% FM"
        out.append(
            f'<tr{hi}><td class="sys">{r["label"]}</td>'
            + cell(r["fm"], bfm) + cell(r["l1"], bl1)
            + f'<td class="num">{delta}</td></tr>'
        )
    out += ["</tbody>", "</table>",
            '<p class="mh-note">One run each, 50 epochs, seed 1234. Lower is better in both loss columns, so a '
            'negative change is an improvement. Both are training objectives, not perceptual scores.</p>', "</div>"]
    (SNIP / "table_main.html").write_text("\n".join(out) + "\n")
    return sel


def table_all(rows):
    """Every logged run, grouped by family."""
    bfm, bl1 = min(r["fm"] for r in rows), min(r["l1"] for r in rows)
    out = ['<div class="mh-table" markdown="0">', "<table>", "<thead><tr>",
           "<th>Run</th><th>Generator</th><th class=\"num\">Feat. match ↓</th>",
           "<th class=\"num\">L1 spec ↓</th><th class=\"num\">d_state</th>",
           "</tr></thead>", "<tbody>"]
    for fam in FAMILY_ORDER:
        fam_rows = sorted((r for r in rows if r["family"] == fam), key=lambda r: r["fm"])
        if not fam_rows:
            continue
        out.append(f'<tr class="mh-group"><th class="grp" colspan="5">{FAMILY_TITLE[fam]}</th></tr>')
        for r in fam_rows:
            hi = ' class="mh-row"' if fam == "gated" and "Hybrid 1" in r["label"] else ""
            out.append(
                f'<tr{hi}><td class="sys">{r["label"]}</td>'
                f'<td><code>{r["generator_class"]}</code></td>'
                + cell(r["fm"], bfm) + cell(r["l1"], bl1)
                + f'<td class="num">{r["d_state"] or "—"}</td></tr>'
            )
    out += ["</tbody>", "</table>",
            '<p class="mh-note">Final-epoch validation losses, one run each at 50 epochs, seed 1234. '
            'Bold marks the lowest value in the table.</p>', "</div>"]
    (SNIP / "table_all.html").write_text("\n".join(out) + "\n")


def home_stats(rows):
    base = next(r for r in rows if r["label"] == "Baseline HiFi-GAN")
    g64 = next(r for r in rows if r["label"] == "Hybrid 1 gated, Mamba-64")
    g128 = next(r for r in rows if r["label"] == "Hybrid 1 gated, Mamba-128")
    worst = max(rows, key=lambda r: r["fm"])

    def stat(title, label, value, sub=None, cls=""):
        s = f'<div class="mh-stat"><p class="mh-stat__corpus">{title}</p>'
        s += (f'<div class="mh-stat__row"><span class="mh-stat__label">{label}</span>'
              f'<span class="mh-stat__value{cls}">{value}</span></div>')
        if sub:
            s += (f'<div class="mh-stat__row"><span class="mh-stat__label">{sub[0]}</span>'
                  f'<span class="mh-stat__value mh-stat__value--oracle">{sub[1]}</span></div>')
        return s + "</div>"

    out = ['<div class="mh-stats" markdown="0">']
    out.append(stat("Runs", "configurations", str(len({r["generator_class"] for r in rows})),
                    ("training runs", str(len(rows)))))
    out.append(stat("Best feature matching", "gated, Mamba-64", fmt(g64["fm"]),
                    ("baseline", fmt(base["fm"]))))
    out.append(stat("Best L1 spectrogram", "gated, Mamba-128", fmt(g128["l1"]),
                    ("baseline", fmt(base["l1"]))))
    archs = {}
    for r in rows:
        archs.setdefault(r["generator_class"], []).append(r["fm"])
    worse = sum(1 for v in archs.values() if min(v) > base["fm"])
    out.append(stat("Worse than baseline", "architectures", f"{worse} of {len(archs)}",
                    ("worst of them", fmt(worst["fm"]))))
    out.append("</div>")
    (SNIP / "home_stats.html").write_text("\n".join(out) + "\n")


def fig_curves():
    """Validation feature-matching and L1 spectrogram loss over the 50 epochs, for the headline runs."""
    with open(ROOT / "results" / "loss_curves.csv") as f:
        rows = list(csv.DictReader(f))

    series = {
        "Original": ("baseline", "#8b919c", "Baseline"),
        "Hybrid 1 Mamba64 Gated1234": ("accent", "#3355d6", "Gated, Mamba-64"),
        "Hybrid 1 Mamba128 Gated1234": ("mamba", "#23977f", "Gated, Mamba-128"),
        "Mamba Only": ("conv", "#b67b0c", "Mamba only"),
    }

    panels = [
        ("G Feat Match Loss / Valid", "Feature matching loss (validation)"),
        ("G L1 Spec Loss / Valid", "L1 spectrogram loss (validation)"),
    ]

    def curve(loss_name, col):
        pts = []
        for r in rows:
            if r["loss_name"] != loss_name:
                continue
            v = r.get(col, "")
            if v not in ("", "nan", None):
                pts.append((float(r["epoch"]), float(v)))
        return sorted(pts)

    W, H = 760, 332
    PAD_L, PAD_R, PAD_T, PAD_B = 52, 14, 26, 60
    pw = (W - PAD_L - PAD_R - 46) / 2

    svg = [f'<svg class="mh-svg mh-wide" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Validation feature-matching and L1 spectrogram loss over 50 epochs for the baseline, '
           f'the two gated models and the Mamba-only ablation.">']

    for pi, (loss_name, title) in enumerate(panels):
        ox = PAD_L + pi * (pw + 46)
        data = {c: curve(loss_name, c) for c in series}
        data = {c: v for c, v in data.items() if v}
        ys = [y for v in data.values() for _, y in v]
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.08 or 0.01
        lo, hi = max(0, lo - pad), hi + pad
        ph = H - PAD_T - PAD_B

        def X(e):
            return ox + (e - 1) / 49 * pw

        def Y(v):
            return PAD_T + ph - (v - lo) / (hi - lo) * ph

        svg.append(f'<text x="{ox:.0f}" y="16" class="f-ink" fill="#1c2230" font-size="11.5" '
                   f'font-weight="600">{title}</text>')
        # axes
        svg.append(f'<path d="M{ox:.1f} {PAD_T} V {PAD_T + ph}" class="s-line" stroke="#c9ccd3" '
                   f'stroke-width="1" fill="none"/>')
        svg.append(f'<path d="M{ox:.1f} {PAD_T + ph} H {ox + pw:.1f}" class="s-line" stroke="#c9ccd3" '
                   f'stroke-width="1" fill="none"/>')
        for frac in (0, 0.5, 1):
            v = lo + frac * (hi - lo)
            y = Y(v)
            svg.append(f'<path d="M{ox:.1f} {y:.1f} H {ox + pw:.1f}" class="s-grid" stroke="#eceef1" '
                       f'stroke-width="1" fill="none"/>')
            svg.append(f'<text x="{ox - 7:.1f}" y="{y + 3.5:.1f}" text-anchor="end" class="f-muted" '
                       f'fill="#6b7180" font-size="9.5">{v:.2f}</text>')
        for e in (1, 25, 50):
            svg.append(f'<text x="{X(e):.1f}" y="{PAD_T + ph + 15:.0f}" text-anchor="middle" '
                       f'class="f-muted" fill="#6b7180" font-size="9.5">{e}</text>')
        svg.append(f'<text x="{ox + pw / 2:.0f}" y="{PAD_T + ph + 31:.0f}" text-anchor="middle" '
                   f'class="f-muted" fill="#6b7180" font-size="10">epoch</text>')

        for col, pts in data.items():
            cls, hexv, _ = series[col]
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(e):.1f} {Y(v):.1f}" for i, (e, v) in enumerate(pts))
            width = 2.2 if cls == "accent" else 1.5
            svg.append(f'<path d="{d}" class="s-{cls}" stroke="{hexv}" stroke-width="{width}" '
                       f'fill="none" stroke-linejoin="round"/>')

    # legend, centred under both panels
    entries = [(cls, hexv, label) for cls, hexv, label in series.values()]
    widths = [26 + len(label) * 5.7 for _, _, label in entries]
    lx = (W - sum(widths)) / 2
    svg.append(f'<g transform="translate(0 {H - 12})">')
    for (cls, hexv, label), w in zip(entries, widths):
        svg.append(f'<path d="M{lx:.1f} -4 h 16" class="s-{cls}" stroke="{hexv}" stroke-width="2.2" fill="none"/>')
        svg.append(f'<text x="{lx + 21:.1f}" y="0" class="f-ink2" fill="#434a5a" font-size="10">{label}</text>')
        lx += w
    svg.append("</g>")
    svg.append("</svg>")

    html = ['<figure class="mh-fig" markdown="0">', "\n".join(svg),
            "<figcaption>Validation loss over the 50 training epochs. The gated models track the baseline "
            "closely throughout and separate only late; the Mamba-only ablation, with no convolutional branch, "
            "stays clearly worse from the start. Single runs, so the gaps between the close curves are within "
            "what seed variation could produce.</figcaption>", "</figure>"]
    (SNIP / "fig_curves.html").write_text("\n".join(html) + "\n")


if __name__ == "__main__":
    SNIP.mkdir(parents=True, exist_ok=True)
    rows = load()
    table_main(rows)
    table_all(rows)
    home_stats(rows)
    fig_curves()
    print(f"wrote {len(list(SNIP.glob('*.html')))} snippets from {len(rows)} runs")
