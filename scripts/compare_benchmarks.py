#!/usr/bin/env python3
"""
Compare Rust vs Python NeuroRVQ benchmarks.
Reads figures/benchmark_results.csv (Rust) and figures/benchmark_python.csv (Python).
Generates comparison charts and a markdown summary.
"""

import csv, os

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def svg_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def grouped_bar_chart(title, labels, rust_vals, python_vals, rust_errs, python_errs, y_label, path):
    n = len(labels)
    margin_l, margin_r, margin_t, margin_b = 80, 140, 50, 130
    bar_w = 18
    pair_gap = 8
    group_gap = 16
    group_w = bar_w * 2 + pair_gap
    chart_w = n * (group_w + group_gap) + group_gap
    chart_h = 260
    w = margin_l + chart_w + margin_r
    h = margin_t + chart_h + margin_b

    all_vals = [v + e for v, e in zip(rust_vals, rust_errs)] + [v + e for v, e in zip(python_vals, python_errs)]
    max_val = max(all_vals) * 1.15 if all_vals else 1.0

    s = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="system-ui,-apple-system,sans-serif" font-size="12">\n'
    s += f'<rect width="{w}" height="{h}" fill="white"/>\n'
    s += f'<text x="{w/2}" y="28" text-anchor="middle" font-size="15" font-weight="600">{svg_escape(title)}</text>\n'

    # Y gridlines
    for i in range(6):
        frac = i / 5
        y = margin_t + chart_h * (1 - frac)
        val = max_val * frac
        s += f'<line x1="{margin_l}" y1="{y}" x2="{margin_l + chart_w}" y2="{y}" stroke="#e0e0e0" stroke-width="1"/>\n'
        s += f'<text x="{margin_l - 6}" y="{y + 4}" text-anchor="end" fill="#666" font-size="10">{val:.0f}</text>\n'

    # Y label
    ym = margin_t + chart_h / 2
    s += f'<text x="14" y="{ym}" text-anchor="middle" transform="rotate(-90,14,{ym})" fill="#333" font-size="11">{y_label}</text>\n'

    rust_color = "#e8710a"   # orange
    python_color = "#306998"  # python blue

    for i in range(n):
        gx = margin_l + group_gap + i * (group_w + group_gap)

        # Rust bar
        bh_r = (rust_vals[i] / max_val) * chart_h
        y_r = margin_t + chart_h - bh_r
        s += f'<rect x="{gx}" y="{y_r}" width="{bar_w}" height="{bh_r}" fill="{rust_color}" rx="2"/>\n'
        s += f'<text x="{gx + bar_w/2}" y="{y_r - 4}" text-anchor="middle" fill="#333" font-size="8">{rust_vals[i]:.0f}</text>\n'
        # error bar
        if rust_errs[i] > 0:
            eh = (rust_errs[i] / max_val) * chart_h
            cx = gx + bar_w / 2
            s += f'<line x1="{cx}" y1="{y_r - eh}" x2="{cx}" y2="{y_r + eh}" stroke="#333" stroke-width="1"/>\n'

        # Python bar
        px = gx + bar_w + pair_gap
        bh_p = (python_vals[i] / max_val) * chart_h
        y_p = margin_t + chart_h - bh_p
        s += f'<rect x="{px}" y="{y_p}" width="{bar_w}" height="{bh_p}" fill="{python_color}" rx="2"/>\n'
        s += f'<text x="{px + bar_w/2}" y="{y_p - 4}" text-anchor="middle" fill="#333" font-size="8">{python_vals[i]:.0f}</text>\n'
        if python_errs[i] > 0:
            eh = (python_errs[i] / max_val) * chart_h
            cx = px + bar_w / 2
            s += f'<line x1="{cx}" y1="{y_p - eh}" x2="{cx}" y2="{y_p + eh}" stroke="#333" stroke-width="1"/>\n'

        # X label
        lx = gx + group_w / 2
        ly = margin_t + chart_h + 10
        s += f'<text x="{lx}" y="{ly}" text-anchor="start" transform="rotate(45,{lx},{ly})" fill="#333" font-size="10">{svg_escape(labels[i])}</text>\n'

    # Legend
    lx = margin_l + chart_w + 16
    ly = margin_t + 20
    s += f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="{rust_color}" rx="2"/>\n'
    s += f'<text x="{lx + 20}" y="{ly + 11}" fill="#333" font-size="11">Rust (Burn)</text>\n'
    s += f'<rect x="{lx}" y="{ly + 24}" width="14" height="14" fill="{python_color}" rx="2"/>\n'
    s += f'<text x="{lx + 20}" y="{ly + 35}" fill="#333" font-size="11">Python (PyTorch)</text>\n'

    s += '</svg>\n'
    with open(path, 'w') as f:
        f.write(s)
    print(f"Chart: {path}")


def speedup_chart(title, labels, speedups, colors, path):
    n = len(labels)
    margin_l, margin_r, margin_t, margin_b = 80, 20, 50, 130
    bar_w = 36
    gap = 12
    chart_w = n * (bar_w + gap) + gap
    chart_h = 260
    w = margin_l + chart_w + margin_r
    h = margin_t + chart_h + margin_b

    max_val = max(speedups) * 1.15 if speedups else 1.0

    s = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="system-ui,-apple-system,sans-serif" font-size="12">\n'
    s += f'<rect width="{w}" height="{h}" fill="white"/>\n'
    s += f'<text x="{w/2}" y="28" text-anchor="middle" font-size="15" font-weight="600">{svg_escape(title)}</text>\n'

    for i in range(6):
        frac = i / 5
        y = margin_t + chart_h * (1 - frac)
        val = max_val * frac
        s += f'<line x1="{margin_l}" y1="{y}" x2="{margin_l + chart_w}" y2="{y}" stroke="#e0e0e0" stroke-width="1"/>\n'
        s += f'<text x="{margin_l - 6}" y="{y + 4}" text-anchor="end" fill="#666" font-size="10">{val:.2f}x</text>\n'

    # 1.0x reference line
    y1 = margin_t + chart_h * (1 - 1.0 / max_val)
    s += f'<line x1="{margin_l}" y1="{y1}" x2="{margin_l + chart_w}" y2="{y1}" stroke="#e74c3c" stroke-width="1.5" stroke-dasharray="6,3"/>\n'
    s += f'<text x="{margin_l + chart_w + 4}" y="{y1 + 4}" fill="#e74c3c" font-size="10">1.0x (parity)</text>\n'

    ym = margin_t + chart_h / 2
    s += f'<text x="14" y="{ym}" text-anchor="middle" transform="rotate(-90,14,{ym})" fill="#333" font-size="11">Rust / Python ratio</text>\n'

    for i in range(n):
        x = margin_l + gap + i * (bar_w + gap)
        bh = (speedups[i] / max_val) * chart_h
        y = margin_t + chart_h - bh
        c = colors[i % len(colors)]
        s += f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" fill="{c}" rx="3"/>\n'
        s += f'<text x="{x + bar_w/2}" y="{y - 5}" text-anchor="middle" fill="#333" font-size="9" font-weight="600">{speedups[i]:.2f}x</text>\n'
        lx = x + bar_w / 2
        ly = margin_t + chart_h + 10
        s += f'<text x="{lx}" y="{ly}" text-anchor="start" transform="rotate(45,{lx},{ly})" fill="#333" font-size="10">{svg_escape(labels[i])}</text>\n'

    s += '</svg>\n'
    with open(path, 'w') as f:
        f.write(s)
    print(f"Chart: {path}")


def main():
    rust = read_csv("figures/benchmark_results.csv")
    python = read_csv("figures/benchmark_python.csv")

    # Match by label
    py_map = {r['label']: r for r in python}
    paired = []
    for r in rust:
        label = r['label']
        if label in py_map:
            paired.append((r, py_map[label]))

    if not paired:
        print("ERROR: No matching labels between Rust and Python CSVs!")
        return

    labels = [r['label'] for r, _ in paired]
    rust_tok = [float(r['tokenize_mean_ms']) for r, _ in paired]
    rust_tok_err = [float(r['tokenize_std_ms']) for r, _ in paired]
    py_tok = [float(p['tokenize_mean_ms']) for _, p in paired]
    py_tok_err = [float(p['tokenize_std_ms']) for _, p in paired]

    rust_enc = [float(r['encode_mean_ms']) for r, _ in paired]
    rust_enc_err = [float(r['encode_std_ms']) for r, _ in paired]
    py_enc = [float(p['encode_mean_ms']) for _, p in paired]
    py_enc_err = [float(p['encode_std_ms']) for _, p in paired]

    # Ratio = Rust_time / Python_time  (>1 means Python is faster)
    tok_ratio = [r / p if p > 0 else 0 for r, p in zip(rust_tok, py_tok)]
    enc_ratio = [r / p if p > 0 else 0 for r, p in zip(rust_enc, py_enc)]

    modality_colors = []
    for r, _ in paired:
        m = r['modality']
        if m == 'EEG': modality_colors.append('#4285f4')
        elif m == 'ECG': modality_colors.append('#ea4335')
        elif m == 'EMG': modality_colors.append('#34a853')
        else: modality_colors.append('#999')

    # Chart 1: Side-by-side tokenize latency
    grouped_bar_chart(
        "Tokenize Latency: Rust (Burn) vs Python (PyTorch)",
        labels, rust_tok, py_tok, rust_tok_err, py_tok_err,
        "Latency (ms)",
        "figures/compare_tokenize.svg",
    )

    # Chart 2: Side-by-side encode latency
    grouped_bar_chart(
        "Encode Latency: Rust (Burn) vs Python (PyTorch)",
        labels, rust_enc, py_enc, rust_enc_err, py_enc_err,
        "Latency (ms)",
        "figures/compare_encode.svg",
    )

    # Chart 3: Ratio (Rust/Python) — values > 1 mean Python is faster
    speedup_chart(
        "Tokenize Time Ratio (Rust / Python) — lower is better for Rust",
        labels, tok_ratio, modality_colors,
        "figures/compare_ratio.svg",
    )

    # ── Markdown summary ──────────────────────────────────────────────────
    md_path = "figures/benchmark_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Rust vs Python Benchmark Comparison\n\n")
        f.write("**Platform:** Apple M4 Pro, 64 GB RAM, macOS (arm64)  \n")
        f.write("**Rust backend:** NdArray + Rayon  \n")
        f.write(f"**Python:** PyTorch 2.8.0 (CPU)  \n")
        f.write(f"**Iterations:** 10 (after 2 warmup)\n\n")

        f.write("| Configuration | Modality | Rust Tokenize (ms) | Python Tokenize (ms) | Ratio (Rust/Py) |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for (r, p), ratio in zip(paired, tok_ratio):
            f.write(f"| {r['label']} | {r['modality']} | "
                    f"{float(r['tokenize_mean_ms']):.1f} ± {float(r['tokenize_std_ms']):.1f} | "
                    f"{float(p['tokenize_mean_ms']):.1f} ± {float(p['tokenize_std_ms']):.1f} | "
                    f"{ratio:.1f}x |\n")

        f.write("\n### Charts\n\n")
        f.write("![Tokenize Comparison](compare_tokenize.svg)\n\n")
        f.write("![Encode Comparison](compare_encode.svg)\n\n")
        f.write("![Speed Ratio](compare_ratio.svg)\n\n")

    print(f"Summary: {md_path}")
    print()

    # Print summary
    avg_ratio = sum(tok_ratio) / len(tok_ratio)
    print(f"Average Rust/Python ratio: {avg_ratio:.1f}x")
    print(f"  → Python (PyTorch) is {avg_ratio:.1f}x faster than Rust (Burn NdArray)")
    print(f"  → This is expected: PyTorch uses optimized BLAS/MKL kernels + graph fusion")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    main()
