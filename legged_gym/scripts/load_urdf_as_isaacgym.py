# file: load_urdf_as_isaacgym.py
from isaacgym import gymapi

# -------------------------------------------------------
# 1️⃣ Create the simulator
# -------------------------------------------------------
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, 9.81)   # 🚫 Disable gravity

# Use CPU pipeline so Pose Override works in the viewer
sim_params.use_gpu_pipeline = False
sim_params.physx.use_gpu = False   # keep physics on CPU for simplicity

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise RuntimeError("Failed to create simulation")

print("🛰️  Gravity disabled — robot will hover in place.")

# -------------------------------------------------------
# 2️⃣ Load your URDF
# -------------------------------------------------------
asset_root = "/home/nathan_lacour/limx_rl/pointfoot-legged-gym/resources/robots"
asset_file = "WF_TRON1A/urdf/robot.urdf"      # change if needed

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
asset_options.collapse_fixed_joints = True
asset_options.replace_cylinder_with_capsule = True

robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# -------------------------------------------------------
# 3️⃣ Create one environment and actor
# -------------------------------------------------------
env = gym.create_env(sim, gymapi.Vec3(-2, 0, 0), gymapi.Vec3(2, 2, 2), 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 2.0)  # start higher
pose.r = gymapi.Quat(0, 0, 0, 1)

robot_handle = gym.create_actor(env, robot_asset, pose, "tron1", 0, 1)

# -------------------------------------------------------
# 4️⃣ Launch viewer
# -------------------------------------------------------
viewer_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, viewer_props)
if viewer is None:
    raise RuntimeError("Failed to create viewer")

print("\nViewer ready — right‑click the robot ➜ *Pose Override* (CPU pipeline, no gravity).")

# -------------------------------------------------------
# 5️⃣ Simulation loop
# -------------------------------------------------------
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)