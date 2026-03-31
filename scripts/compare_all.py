#!/usr/bin/env python3
"""
Compare all backends: Rust NdArray, Rust wgpu, Python PyTorch.
Reads:
  figures/benchmark_results_ndarray.csv (or benchmark_results.csv)
  figures/benchmark_results_wgpu.csv
  figures/benchmark_python.csv
Generates comparison charts and markdown.
"""

import csv, os

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def svg_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def multi_bar_chart(title, labels, series, y_label, path):
    """
    series: list of (name, color, values, errors)
    """
    n_groups = len(labels)
    n_series = len(series)
    margin_l, margin_r, margin_t, margin_b = 80, 160, 50, 130
    bar_w = 16
    pair_gap = 3
    group_w = n_series * bar_w + (n_series - 1) * pair_gap
    group_gap = 14
    chart_w = n_groups * (group_w + group_gap) + group_gap
    chart_h = 260
    w = margin_l + chart_w + margin_r
    h = margin_t + chart_h + margin_b

    all_tops = []
    for _, _, vals, errs in series:
        for v, e in zip(vals, errs):
            if v is not None:
                all_tops.append(v + e)
    max_val = max(all_tops) * 1.15 if all_tops else 1.0

    s = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="system-ui,-apple-system,sans-serif" font-size="12">\n'
    s += f'<rect width="{w}" height="{h}" fill="white"/>\n'
    s += f'<text x="{w/2}" y="28" text-anchor="middle" font-size="15" font-weight="600">{svg_escape(title)}</text>\n'

    for i in range(6):
        frac = i / 5
        y = margin_t + chart_h * (1 - frac)
        val = max_val * frac
        s += f'<line x1="{margin_l}" y1="{y}" x2="{margin_l + chart_w}" y2="{y}" stroke="#e0e0e0" stroke-width="1"/>\n'
        s += f'<text x="{margin_l - 6}" y="{y + 4}" text-anchor="end" fill="#666" font-size="10">{val:.0f}</text>\n'

    ym = margin_t + chart_h / 2
    s += f'<text x="14" y="{ym}" text-anchor="middle" transform="rotate(-90,14,{ym})" fill="#333" font-size="11">{y_label}</text>\n'

    for gi in range(n_groups):
        gx = margin_l + group_gap + gi * (group_w + group_gap)

        for si, (sname, color, vals, errs) in enumerate(series):
            bx = gx + si * (bar_w + pair_gap)
            v = vals[gi]
            e = errs[gi]
            if v is None:
                continue
            bh = (v / max_val) * chart_h
            y = margin_t + chart_h - bh
            s += f'<rect x="{bx}" y="{y}" width="{bar_w}" height="{bh}" fill="{color}" rx="2"/>\n'
            s += f'<text x="{bx + bar_w/2}" y="{y - 3}" text-anchor="middle" fill="#333" font-size="7">{v:.0f}</text>\n'
            if e > 0:
                eh = (e / max_val) * chart_h
                cx = bx + bar_w / 2
                s += f'<line x1="{cx}" y1="{y - eh}" x2="{cx}" y2="{y + eh}" stroke="#333" stroke-width="1"/>\n'

        lx = gx + group_w / 2
        ly = margin_t + chart_h + 10
        s += f'<text x="{lx}" y="{ly}" text-anchor="start" transform="rotate(45,{lx},{ly})" fill="#333" font-size="10">{svg_escape(labels[gi])}</text>\n'

    # Legend
    lx = margin_l + chart_w + 16
    ly = margin_t + 10
    for si, (sname, color, _, _) in enumerate(series):
        yy = ly + si * 22
        s += f'<rect x="{lx}" y="{yy}" width="14" height="14" fill="{color}" rx="2"/>\n'
        s += f'<text x="{lx + 20}" y="{yy + 11}" fill="#333" font-size="11">{svg_escape(sname)}</text>\n'

    s += '</svg>\n'
    with open(path, 'w') as f:
        f.write(s)
    print(f"  Chart: {path}")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

    # Load all data
    ndarray = read_csv("figures/benchmark_results_ndarray.csv") or read_csv("figures/benchmark_results.csv")
    wgpu = read_csv("figures/benchmark_results_wgpu.csv")
    python = read_csv("figures/benchmark_python.csv")

    nd_map = {r['label']: r for r in ndarray}
    wgpu_map = {r['label']: r for r in wgpu}
    py_map = {r['label']: r for r in python}

    # Union of all labels, preserving order
    all_labels = []
    seen = set()
    for src in [ndarray, wgpu, python]:
        for r in src:
            if r['label'] not in seen:
                all_labels.append(r['label'])
                seen.add(r['label'])

    def get_val(m, label, field):
        if label in m:
            return float(m[label][field])
        return None

    nd_tok  = [get_val(nd_map, l, 'tokenize_mean_ms') for l in all_labels]
    nd_err  = [get_val(nd_map, l, 'tokenize_std_ms') or 0 for l in all_labels]
    wg_tok  = [get_val(wgpu_map, l, 'tokenize_mean_ms') for l in all_labels]
    wg_err  = [get_val(wgpu_map, l, 'tokenize_std_ms') or 0 for l in all_labels]
    py_tok  = [get_val(py_map, l, 'tokenize_mean_ms') for l in all_labels]
    py_err  = [get_val(py_map, l, 'tokenize_std_ms') or 0 for l in all_labels]

    nd_enc  = [get_val(nd_map, l, 'encode_mean_ms') for l in all_labels]
    nd_enc_err = [get_val(nd_map, l, 'encode_std_ms') or 0 for l in all_labels]
    wg_enc  = [get_val(wgpu_map, l, 'encode_mean_ms') for l in all_labels]
    wg_enc_err = [get_val(wgpu_map, l, 'encode_std_ms') or 0 for l in all_labels]
    py_enc  = [get_val(py_map, l, 'encode_mean_ms') for l in all_labels]
    py_enc_err = [get_val(py_map, l, 'encode_std_ms') or 0 for l in all_labels]

    series_tok = [
        ("Rust NdArray (CPU)", "#e8710a", nd_tok, nd_err),
        ("Rust wgpu (GPU)", "#8e44ad", wg_tok, wg_err),
        ("Python PyTorch (CPU)", "#306998", py_tok, py_err),
    ]
    series_enc = [
        ("Rust NdArray (CPU)", "#e8710a", nd_enc, nd_enc_err),
        ("Rust wgpu (GPU)", "#8e44ad", wg_enc, wg_enc_err),
        ("Python PyTorch (CPU)", "#306998", py_enc, py_enc_err),
    ]

    print("Generating comparison charts...")
    multi_bar_chart(
        "Tokenize Latency: NdArray vs wgpu vs PyTorch",
        all_labels, series_tok, "Latency (ms)",
        "figures/compare_all_tokenize.svg",
    )
    multi_bar_chart(
        "Encode Latency: NdArray vs wgpu vs PyTorch",
        all_labels, series_enc, "Latency (ms)",
        "figures/compare_all_encode.svg",
    )

    # Markdown
    md_path = "figures/benchmark_comparison.md"
    with open(md_path, 'w') as f:
        f.write("# Benchmark Comparison: Rust (NdArray / wgpu) vs Python (PyTorch)\n\n")
        f.write("**Platform:** Apple M4 Pro, 64 GB RAM, macOS (arm64)  \n")
        f.write("**Rust backends:** Burn NdArray+Rayon (CPU), Burn wgpu (GPU)  \n")
        f.write("**Python:** PyTorch 2.8.0 (CPU, Apple Accelerate)  \n")
        f.write("**Iterations:** 10 (after 2 warmup)\n\n")

        f.write("| Configuration | Modality | NdArray (ms) | wgpu (ms) | PyTorch (ms) | wgpu vs PyTorch |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for i, label in enumerate(all_labels):
            mod = nd_map.get(label, wgpu_map.get(label, py_map.get(label, {})))
            modality = mod.get('modality', '?')

            def fmt(v, e):
                if v is None:
                    return "—"
                return f"{v:.0f} ± {e:.0f}"

            nd_s = fmt(nd_tok[i], nd_err[i])
            wg_s = fmt(wg_tok[i], wg_err[i])
            py_s = fmt(py_tok[i], py_err[i])

            ratio_s = "—"
            if wg_tok[i] is not None and py_tok[i] is not None and py_tok[i] > 0:
                ratio = wg_tok[i] / py_tok[i]
                ratio_s = f"{ratio:.1f}x"

            f.write(f"| {label} | {modality} | {nd_s} | {wg_s} | {py_s} | {ratio_s} |\n")

        f.write("\n### Charts\n\n")
        f.write("![Tokenize Comparison](compare_all_tokenize.svg)\n\n")
        f.write("![Encode Comparison](compare_all_encode.svg)\n\n")

        # Summary
        f.write("### Analysis\n\n")

        # Compute avg ratios where data exists
        wgpu_py_ratios = []
        nd_py_ratios = []
        wgpu_nd_ratios = []
        for i in range(len(all_labels)):
            if wg_tok[i] and py_tok[i] and py_tok[i] > 0:
                wgpu_py_ratios.append(wg_tok[i] / py_tok[i])
            if nd_tok[i] and py_tok[i] and py_tok[i] > 0:
                nd_py_ratios.append(nd_tok[i] / py_tok[i])
            if wg_tok[i] and nd_tok[i] and nd_tok[i] > 0:
                wgpu_nd_ratios.append(nd_tok[i] / wg_tok[i])

        if wgpu_py_ratios:
            avg = sum(wgpu_py_ratios) / len(wgpu_py_ratios)
            f.write(f"- **wgpu vs PyTorch:** Rust wgpu is {avg:.1f}x {'slower' if avg > 1 else 'faster'} than PyTorch on average\n")
        if nd_py_ratios:
            avg = sum(nd_py_ratios) / len(nd_py_ratios)
            f.write(f"- **NdArray vs PyTorch:** Rust NdArray is {avg:.1f}x slower than PyTorch on average\n")
        if wgpu_nd_ratios:
            avg = sum(wgpu_nd_ratios) / len(wgpu_nd_ratios)
            f.write(f"- **wgpu vs NdArray:** wgpu is {avg:.1f}x faster than NdArray (CPU→GPU speedup)\n")

        f.write(f"\n**Note:** ECG configs are skipped on wgpu due to a burn-wgpu shared memory limitation with small embed_dim (40).\n")

    print(f"  Summary: {md_path}")

    # Print summary
    if wgpu_nd_ratios:
        avg = sum(wgpu_nd_ratios) / len(wgpu_nd_ratios)
        print(f"\n  wgpu is {avg:.1f}x faster than NdArray (GPU vs CPU)")
    if wgpu_py_ratios:
        avg = sum(wgpu_py_ratios) / len(wgpu_py_ratios)
        print(f"  PyTorch is {avg:.1f}x faster than wgpu")


if __name__ == "__main__":
    main()
