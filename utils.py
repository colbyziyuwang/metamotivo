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
    Return python slice for the requested body - component in the flattened obs.

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
        return slice(start, start + ANG_PER_B - 1)

    raise ValueError("kind must be one of 'pos', 'rot', 'vel', 'ang'")

# ------------- quick sanity check -------------
if __name__ == "__main__":
    # expect 358
    assert OBS_DIM == 358, OBS_DIM

    # root rot slice should be 70-75
    print("Root rotation slice:", slice_for("Pelv", "rot"))
    # L_Hip position slice should start at OFF_POS (1) because it's the first link
    print("L_Hip pos slice:", slice_for("L_Hip", "pos"))
