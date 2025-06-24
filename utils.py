def get_joint_idx(name):
    # Load text file
    file = "joint_names.txt"
    with open(file, "r") as f:
        lines = f.readlines()
    # Remove newline characters
    lines = [line.strip() for line in lines]
    # Find index of name
    try:
        idx = lines.index(name)
    except ValueError:
        raise ValueError(f"Joint name '{name}' not found in {file}.")
    return idx

def get_joint_name(idx):
    # Load text file
    file = "joint_names.txt"
    with open(file, "r") as f:
        lines = f.readlines()
    # Remove newline characters
    lines = [line.strip() for line in lines]
    # Check if index is valid
    if idx < 0 or idx >= len(lines):
        raise IndexError(f"Index {idx} out of range for {file}.")
    return lines[idx]

# print(get_joint_name(2))
# print(get_joint_idx("L_Knee"))