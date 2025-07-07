# log2latex.py
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple, List

# --------------------------------------------
# regex helpers
_TASK_RE   = re.compile(r"🎯\s*Task:\s*(.+)")
_LINE_RE   = re.compile(r"([^\s:]+):\s*([-\d.]+)\s*±\s*([-\d.]+)")

def _parse_block(block: List[str]) -> Tuple[str, Dict[str, str]]:
    """
    block -> ("move-ego-0-0", {"Reward": "257.58 ± 8.77", "Cost": "2678.11 ± 72.97", ...})
    """
    task_match = _TASK_RE.match(block[0])
    if not task_match:
        return None, None
    task_name = task_match.group(1).strip()

    stats = OrderedDict()          # Reward / Cost / (optional) Q_c
    for ln in block[1:]:
        m = _LINE_RE.search(ln)
        if m:
            label, mean, std = m.groups()
            stats[label] = f"{mean} $\\pm$ {std}"
    return task_name, stats

def read_log(path: str | Path) -> Dict[str, Dict[str, str]]:
    """
    Returns OrderedDict  task_name -> {Reward: ".. ± ..", Cost: ".. ± ..", (Q_c: ".. ± ..")}
    """
    text = Path(path).read_text(encoding="utf-8").splitlines()

    blocks, current = [], []
    for ln in text:
        if ln.strip() == "":                 # blank → block boundary
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(ln)
    if current:                              # trailing block
        blocks.append(current)

    table = OrderedDict()
    for blk in blocks:
        task, stats = _parse_block(blk)
        if task is None or stats is None:
            continue
        table[task] = stats
    return table

def make_latex(table: Dict[str, Dict[str, str]]) -> str:
    # decide columns   (Reward, Cost, maybe Q_c)
    header_labels = list(next(iter(table.values())).keys())   # look at first task
    col_spec = "l" + "c" * len(header_labels)

    lines = [
        f"\\begin{{tabular}}{{|{col_spec}|}}",
        "\\hline",
        "Task & " + " & ".join(header_labels) + r" \\",
        "\\hline"
    ]
    for task, stats in table.items():
        row = [task] + [stats[k] for k in header_labels]
        lines.append(" & ".join(row) + r" \\")
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines)

# ------------------- quick CLI -------------------
if __name__ == "__main__":
    import argparse, sys
    pa = argparse.ArgumentParser(description="Convert Metamotivo log to LaTeX table.")
    pa.add_argument("logfile", help="path to *.txt produced by your rollout script")
    args = pa.parse_args()

    tbl  = read_log(args.logfile)
    code = make_latex(tbl)
    print(code)
