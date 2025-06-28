import re
from pathlib import Path
from typing import Tuple, List
import torch
import numpy as np
import random

# -------------------------------------------
# obs layout constants (HumEnv – SMPL humanoid)
# -------------------------------------------
DIM_H       = 1                     # root height
POS_PER_B   = 3                     # xyz in local frame       (root excluded ⇒ 23 bodies)
ROT_PER_B   = 6                     # tan-norm quaternion      (all 24 bodies)
VEL_PER_B   = 3                     # linear velocity          (all 24 bodies)
ANG_PER_B   = 3                     # angular velocity         (all 24 bodies)

N_BODY      = 24                    # root + 23 links

LEN_POS     = 23 * POS_PER_B        # 69
LEN_ROT     = N_BODY * ROT_PER_B    # 144
LEN_VEL     = N_BODY * VEL_PER_B    # 72
LEN_ANG     = N_BODY * ANG_PER_B    # 72

# section offsets in the flattened observation (root_h | pos | rot | vel | ang)
OFF_H       = 0
OFF_POS     = OFF_H   + DIM_H                 # 1
OFF_ROT     = OFF_POS + LEN_POS              # 70
OFF_VEL     = OFF_ROT + LEN_ROT              # 214
OFF_ANG     = OFF_VEL + LEN_VEL              # 286
OBS_DIM     = OFF_ANG + LEN_ANG              # 358

# -------------------------------------------------
# body name list – order matches MuJoCo body id’s
# -------------------------------------------------
BODY_NAMES = [
    "Pelv", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

NAME2IDX = {name: i for i, name in enumerate(BODY_NAMES)}

# ----------------------------------------------------------------
# helper ----------------------------------------------------------------
def set_seed(SEED):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_body_idx(name: str) -> int:
    try:
        return NAME2IDX[name]
    except KeyError:
        raise ValueError(f"Unknown body name: {name!r}. Allowed: {BODY_NAMES}")

def get_body_name(idx: int) -> str:
    if 0 <= idx < N_BODY:
        return BODY_NAMES[idx]
    raise IndexError(f"Body index {idx} out of range 0-{N_BODY-1}")

def slice_for(body: str | int, kind: str) -> slice:
    """
    Return start and end indices (inclusive) for the requested body - component in the flattened obs.

    kind ∈ {"pos", "rot", "vel", "ang"}
    """
    if isinstance(body, str):
        b_id = get_body_idx(body)
    else:
        b_id = body
    if not (0 <= b_id < N_BODY):
        raise IndexError("Body id out of range")

    if kind == "pos":
        if b_id == 0:                       # root position is *not* stored in local_body_pos
            raise ValueError("Root has no local_body_pos entry")
        start = OFF_POS + (b_id - 1) * POS_PER_B
        return start, start + POS_PER_B - 1

    if kind == "rot":
        start = OFF_ROT + b_id * ROT_PER_B
        return start, start + ROT_PER_B - 1

    if kind == "vel":
        start = OFF_VEL + b_id * VEL_PER_B
        return start, start + VEL_PER_B - 1

    if kind == "ang":
        start = OFF_ANG + b_id * ANG_PER_B
        return start, start + ANG_PER_B - 1

    raise ValueError("kind must be one of 'pos', 'rot', 'vel', 'ang'")

STATS_PATH = Path(__file__).with_name("metamotivo_observation_stats.txt")

_DIM_LINE_RE = re.compile(
    r"^\s*Dimension\s+(?P<idx>\d+):"
    r"\s*min=(?P<min>-?\d+\.\d+),\s*max=(?P<max>-?\d+\.\d+),"
    r"\s*mean=(?P<mean>-?\d+\.\d+),\s*std=(?P<std>-?\d+\.\d+)"
)

def _read_task_block(task: str, fpath: Path = STATS_PATH) -> List[str]:
    """Return the lines belonging to one `Task:` section."""
    if not fpath.exists():
        raise FileNotFoundError(f"stats file not found at {fpath!s}")

    lines_for_task: List[str] = []
    with fpath.open("r") as fh:
        in_block = False
        for line in fh:
            if line.startswith("Task:"):
                in_block = line.strip().split("Task:")[1].strip() == task
                continue
            if in_block and (line.strip().startswith("Dimension") or not line.strip()):
                # keep dimension lines; stop when we hit a blank followed by next task
                if line.strip():
                    lines_for_task.append(line.rstrip())
                else:
                    break
    if not lines_for_task:
        raise ValueError(f"Task {task!r} not found in stats file {fpath.name}")
    return lines_for_task

def _parse_dim_lines(dim_lines: List[str]) -> dict[int, dict[str, float]]:
    """Convert 'Dimension k: min=…, …' lines into a mapping."""
    out = {}
    for ln in dim_lines:
        m = _DIM_LINE_RE.match(ln)
        if not m:
            continue
        idx = int(m["idx"])
        out[idx] = {k: float(m[k]) for k in ("min", "max", "mean", "std")}
    return out

def suggest_constraint_range(
    task: str,
    body: str | int,
    kind: str,
    stats_path: Path | str = STATS_PATH,
) -> Tuple[float, float]:
    """
    Heuristically pick a [lower, upper] constraint interval (inclusive) for
    one body-component, based on observation statistics.

    Parameters
    ----------
    task        : name in your stats file ("move-ego-0-0", …)
    body        : body name ("L_Hip") *or* numeric id (0-23)
    kind        : "pos", "rot", "vel" or "ang"
    stats_path  : location of the *.txt with the statistics

    Returns
    -------
    (lo, hi)    : floats – proposed inclusive bounds
    """
    # 1. which dimensions does this body-component occupy?
    start, end = slice_for(body, kind)
    dim_idx_set = set(range(start, end + 1))

    # 2. fetch the stats for that task
    block_lines = _read_task_block(task, Path(stats_path))
    dim_stats = _parse_dim_lines(block_lines)

    # 3. collect candidate bounds
    lows, highs = [], []
    for d in dim_idx_set:
        s = dim_stats.get(d)
        if s is None:
            raise KeyError(f"Dimension {d} not found for task {task!r}. "
                           f"(Check that stats are complete.)")
        # Get min and max
        lows.append(s["min"])
        highs.append(s["max"])

    lo, hi = max(lows), min(highs) # max of mins, min of maxes
    if (lo > hi):
        # switch the range
        lo, hi = hi, lo

    return dim_idx_set, lo, hi

# ------------- quick sanity check -------------
if __name__ == "__main__":
    # expect 358
    assert OBS_DIM == 358, OBS_DIM

    # root rot slice should be 70-75
    print("Root rotation slice:", slice_for("Pelv", "rot"))
    # L_Hip position slice should start at OFF_POS (1) because it's the first link
    print("L_Hip pos slice:", slice_for("L_Hip", "pos"))

    # call the function to suggest constraint range
    dim_idx_set, lo, hi = suggest_constraint_range(
        "move-ego-0-0", "L_Hip", "pos", stats_path=STATS_PATH
    )
    print(f"Dimensions for L_Hip pos: {dim_idx_set}")
    print(f"Suggested constraint range for L_Hip pos: [{lo}, {hi}]")
