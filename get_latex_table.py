import re
from collections import defaultdict

def parse_log_file(file_path):
    with open(file_path, "r") as f:
        log_text = f.read()

    pattern = re.compile(
        r"task1: (.+?)\ntask2: (.+?)\n"
        r"reward_z1_env1: ([\d.]+) ± ([\d.]+)\n"
        r"reward_z1_env2: ([\d.]+) ± ([\d.]+)\n"
        r"reward_iip_env1: ([\d.]+) ± ([\d.]+)\n"
        r"reward_iip_env2: ([\d.]+) ± ([\d.]+)",
        re.MULTILINE
    )

    task_dict = defaultdict(list)
    for match in pattern.findall(log_text):
        task1, task2, *metrics = match
        task_dict[task1].append((task2, metrics))

    return task_dict

def generate_latex_table(task_dict):
    lines = []
    lines.append(r"\begin{tabular}{|l|l|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"Task 1 & Task 2 & $R_{z_1}$ (env1) & $R_{z_1}$ (env2) & $R_{\text{IIP}}$ (env1) & $R_{\text{IIP}}$ (env2) \\")
    lines.append(r"\hline")

    for task1, task2_entries in task_dict.items():
        for i, (task2, metrics) in enumerate(task2_entries):
            r1_mean, r1_std, r2_mean, r2_std, i1_mean, i1_std, i2_mean, i2_std = metrics
            task1_cell = task1 if i == 0 else ""
            row = f"{task1_cell} & {task2} & {r1_mean} ± {r1_std} & {r2_mean} ± {r2_std} & {i1_mean} ± {i1_std} & {i2_mean} ± {i2_std} \\\\"
            lines.append(row)
            lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    return "\n".join(lines)

if __name__ == "__main__":
    input_file = "auto_IIP_eval_per_seed_threshold_bisection.txt"
    task_data = parse_log_file(input_file)
    latex_code = generate_latex_table(task_data)
    table_name = "latex_table_output.tex"
    with open(table_name, "w") as out:
        out.write(latex_code)
    print("✅ LaTeX table written to latex_table_output.tex")
