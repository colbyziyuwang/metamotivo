# log2latex.py
import re, argparse
from pathlib import Path
from collections import OrderedDict, defaultdict

# Example usage
"""
python log2latex.py \
>    "lagrange_output_L_Hip_vel (gradient descent).txt" \
>    "lagrange_output_L_Hip_vel (baseline).txt" \
>    --labels "GradDesc" "Baseline" > combined_table.tex
"""

# -------- regex --------------
TASK_RE  = re.compile(r"🎯\s*Task:\s*(.+)")
LINE_RE  = re.compile(r"([A-Za-z_]+):\s*([-\d.]+)\s*±\s*([-\d.]+)")
LABEL_MAP = {"Reward": "Reward", "Cost": "Cost", "Q_c": "$Q_c$"}

def parse_file(path: Path) -> OrderedDict:
    """
    Returns OrderedDict{task -> {metric -> 'mean ± std'}}
    """
    rows = OrderedDict()
    block = []
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if line.strip() == "":
            if block:
                task_m, data = _parse_block(block)
                if task_m:
                    rows[task_m] = data
            block = []
        else:
            block.append(line)
    return rows

def _parse_block(lines):
    m = TASK_RE.match(lines[0])
    if not m:
        return None, None
    task = m.group(1).strip()
    stats = {}
    for ln in lines[1:]:
        mm = LINE_RE.search(ln)
        if mm:
            key, mean, std = mm.groups()
            stats[key] = f"{mean} $\\pm$ {std}"
    return task, stats


def merge_tables(files, labels):
    """
    -> mapping  task -> {label -> {metric -> text}}
    """
    big = defaultdict(lambda: defaultdict(dict))
    for f, lab in zip(files, labels):
        parsed = parse_file(Path(f))
        for task, metrics in parsed.items():
            for mlabel, val in metrics.items():
                big[task][lab][LABEL_MAP[mlabel]] = val
    return big


def to_latex(big, labels):
    metrics = ["Reward", "Cost"]
    has_qc = any("$Q_c$" in d for task in big.values()
                                 for d in task.values())
    if has_qc:
        metrics.append("$Q_c$")

    header_cols = [f"{lab} {m}" for lab in labels for m in metrics]
    col_spec = "l" + "c" * len(header_cols)

    out  = [f"\\begin{{tabular}}{{|{col_spec}|}}",
            "\\hline",
            "Task & " + " & ".join(header_cols) + r" \\",
            "\\hline"]
    for task in big:
        cells = [task]
        for lab in labels:
            for m in metrics:
                cells.append(big[task][lab].get(m, "--"))
        out.append(" & ".join(cells) + r" \\")
    out += ["\\hline", "\\end{tabular}"]
    return "\n".join(out)


# ---------------- CLI -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="log files to merge (≥2)")
    ap.add_argument("--labels", nargs="+",
                    help="column-group labels (defaults = stem of file names)")
    args = ap.parse_args()

    labs = args.labels or [Path(f).stem for f in args.logs]
    if len(labs) != len(args.logs):
        ap.error("--labels must match number of log files")

    tbl = merge_tables(args.logs, labs)
    latex = to_latex(tbl, labs)
    print(latex)
