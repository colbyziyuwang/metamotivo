import gymnasium
from humenv import make_humenv

def get_inner_mujoco_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env

# Create and unwrap
env, _ = make_humenv(
    num_envs=1,
    task="move-ego-0-0",
    state_init="DefaultAndFall",
    wrappers=[gymnasium.wrappers.FlattenObservation]
)

mj_env = get_inner_mujoco_env(env)

# Debug checks
print("Class:", type(mj_env))
print("Has .model?", hasattr(mj_env, "model"))
print("Has .data?", hasattr(mj_env, "data"))

# # All (public) attribute & method names ---------------------------------
# attrs = [a for a in dir(mj_env.model) if not a.startswith("_")]
# print("MjModel attributes/methods (" + str(len(attrs)) + "):")
# print(fill("  ".join(sorted(attrs)), width=100))

model = mj_env.model

# All joint names
n_joints = model.njnt
print(f"Total joints: {n_joints}")

names = []
# Use names from name_jntadr + names
for i in range(n_joints):
    name_offset = model.name_jntadr[i]
    name = model.names[name_offset:].split(b'\x00', 1)[0].decode()
    qpos_adr = model.jnt_qposadr[i]
    # print(f"[{i}] Joint: {name} → qpos index: {qpos_adr}")
    names.append(name[0:(len(name) - 2)]) # Remove last 2 characters

# Remove duplicat entries from names
names = list(dict.fromkeys(names))
print(names)

# Store list into a file
with open("body_names.txt", "w") as f:
    for name in names:
        f.write(name + "\n")