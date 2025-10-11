import numpy as np
import matplotlib.pyplot as plt
from isaacgym import terrain_utils
from legged_gym.envs.wheelfoot_flat.wheelfoot_flat_config import BipedCfgWF
from legged_gym.utils.terrain import Terrain

# --- 1️⃣ Generate terrain in Python using the same Isaac logic ---
cfg = BipedCfgWF.terrain
terrain = Terrain(cfg, num_robots=1)  # or set num_robots as needed
heightfield = terrain.height_field_raw  # or terrain.heightsamples if you want 1D


# --- 2️⃣ Normalize and export to grayscale PNG ---
height_min, height_max = heightfield.min(), heightfield.max()
normalized = (heightfield - height_min) / (height_max - height_min)
plt.imsave("terrain.png", normalized, cmap="gray")

# --- 3️⃣ Generate .world file for Gazebo automatically ---
world_text = f"""
<sdf version="1.6">
  <world name="isaac_terrain">
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <model name="isaac_terrain">
      <static>true</static>
      <link name="terrain_link">
        <collision name="terrain_collision">
          <geometry>
            <heightmap>
              <uri>file://$(pwd)/terrain.png</uri>
              <size>100 100 10</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
        </collision>
        <visual name="terrain_visual">
          <geometry>
            <heightmap>
              <uri>file://$(pwd)/terrain.png</uri>
              <size>100 100 10</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
"""

with open("isaac_terrain.world", "w") as file:
    file.write(world_text)
print("✅ Created isaac_terrain.world for Gazebo.")