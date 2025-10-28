# file: test_initial_state_in_isaacgym.py
from isaacgym import gymapi
import numpy as np

# -------------------------------------------------------
# 1️⃣  Create the simulator
# -------------------------------------------------------
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)   # normal gravity
sim_params.use_gpu_pipeline = False                 # CPU pipeline = Pose Override OK
sim_params.physx.use_gpu = False

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise RuntimeError("❌ Failed to create simulation")

# -------------------------------------------------------
# 2️⃣  Add a ground plane
# -------------------------------------------------------
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0
gym.add_ground(sim, plane_params)
print("✅ Ground plane added (z = 0)")

# -------------------------------------------------------
# 3️⃣  Describe robot initial state (no class needed)
# -------------------------------------------------------
pos = [0.0, 0.0, 0.5]                  # x, y, z [m]
# 90° pitch forward (around Y)
rot = [0.0, 0.441, 0.0, 0.898]            # x, y, z, w quaternion
lin_vel = [0.0, 0.0, 0.0]              # linear velocity
ang_vel = [0.0, 0.0, 0.0]              # angular velocity
safety_parameter = 0.95

default_joint_angles = {
    "abad_L_Joint": 0.1543,
    "hip_L_Joint": -1.0124,
    "knee_L_Joint": 1.3614,
    "foot_L_Joint": 0.0,
    "abad_R_Joint": -0.1543,
    "hip_R_Joint": 1.0124,
    "knee_R_Joint": -1.3614,
    "foot_R_Joint": 0.0,
    "wheel_L_Joint": 1.8689,
    "wheel_R_Joint": 1.8689,

    # "abad_L_Joint": 0.1543,
    # "hip_L_Joint": -1.0124/2,
    # "knee_L_Joint": 1.3614,
    # "foot_L_Joint": 0.0,
    # "abad_R_Joint": -0.1543,
    # "hip_R_Joint": 1.0124/2,
    # "knee_R_Joint": -1.3614,
    # "foot_R_Joint": 0.0,
    # "wheel_L_Joint": 1.8689,
    # "wheel_R_Joint": 1.8689,
}

asset_root = "/home/nathan_lacour/limx_rl/pointfoot-legged-gym/resources/robots/WF_TRON1A"
asset_file = "urdf/robot.urdf"

asset_opt = gymapi.AssetOptions()
asset_opt.fix_base_link = False
asset_opt.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
asset_opt.collapse_fixed_joints = True
asset_opt.replace_cylinder_with_capsule = True

robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_opt)
if robot_asset is None:
    raise RuntimeError(f"❌ Failed to load robot asset at {asset_root}/{asset_file}")
else:
    print("✅ Robot asset loaded successfully.")

env = gym.create_env(sim, gymapi.Vec3(-2, 0, 0), gymapi.Vec3(2, 2, 2), 1)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(*pos)
pose.r = gymapi.Quat(*rot)
robot_handle = gym.create_actor(env, robot_asset, pose, "tron1", 0, 1)

# ---- Joint setup (compatible fix) ----
dof_names = gym.get_asset_dof_names(robot_asset)
dof_dict = {n: i for i, n in enumerate(dof_names)}
num_dofs = len(dof_names)

dtype = gymapi.DofState.dtype
dof_states = np.zeros(num_dofs, dtype=dtype)  # pos+vel fields
for jname, target in default_joint_angles.items():
    if jname in dof_dict:
        dof_states["pos"][dof_dict[jname]] = target
gym.set_actor_dof_states(env, robot_handle, dof_states, gymapi.STATE_ALL)
# After your loop that fills dof_states["pos"]
target_positions = np.zeros(num_dofs, dtype=np.float32)
for jname, target in default_joint_angles.items():
    if jname in dof_dict:
        target_positions[dof_dict[jname]] = target

gym.set_actor_dof_position_targets(env, robot_handle, target_positions)
# # -------------------------------------------------------
# # 7️⃣  (Optional) set initial linear & angular velocity
# # -------------------------------------------------------
# state = gym.get_actor_rigid_body_states(env, robot_handle, gymapi.STATE_ALL)

# lin = np.array(lin_vel, dtype=np.float32)
# ang = np.array(ang_vel, dtype=np.float32)
# names = state.dtype.names

# # Each branch writes the same 3‑vector to every rigid‑body row -----------------
# if "linvel" in names and "angvel" in names:
#     state["linvel"][:] = lin
#     state["angvel"][:] = ang

# elif "linear" in names and "angular" in names:
#     state["linear"][:] = lin
#     state["angular"][:] = ang

# elif "vel" in names:
#     # Handle compound field variants
#     vfield = state["vel"]
#     # case: compound record with 'linear'/'angular' fields
#     if vfield.dtype.names and "linear" in vfield.dtype.names and "angular" in vfield.dtype.names:
#         n = vfield["linear"].shape[0]
#         vfield["linear"][:] = np.tile(lin, (n, 1))
#         vfield["angular"][:] = np.tile(ang, (n, 1))
#     # case: flattened (N, 6)
#     elif vfield.ndim == 2 and vfield.shape[1] >= 6:
#         vfield[:, 0:3] = lin
#         vfield[:, 3:6] = ang
#     else:
#         print(f"⚠️  Unknown inner structure for field 'vel': shape={vfield.shape}, dtype={vfield.dtype}")
# else:
#     print(f"⚠️  Unrecognized rigid‑body‑state fields: {names}")
#     print("    Leaving velocities unchanged (default = 0).")

# gym.set_actor_rigid_body_states(env, robot_handle, state, gymapi.STATE_ALL)

# -------------------------------------------------------
# 8️⃣  Launch viewer
# -------------------------------------------------------
viewer_props = gymapi.CameraProperties()
viewer_props.width = 1280
viewer_props.height = 720
viewer = gym.create_viewer(sim, viewer_props)
if viewer is None:
    raise RuntimeError("Failed to create viewer")

print("\n🎮  Viewer ready — experiment with base position, quaternion, or joint targets!")

# -------------------------------------------------------
# 9️⃣  Simulation loop
# -------------------------------------------------------
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)