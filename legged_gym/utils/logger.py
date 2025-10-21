# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# [license text unchanged for brevity]

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None
        self.plot_reward_process = None   # ➕ new handle for rewards plot

    # ------------------------------------------------------------
    # Logging functions
    # ------------------------------------------------------------
    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        # Log per-step rewards — assumes dict contains reward components
        for key, value in dict.items():
            if "rew" in key:
                # store scalar float for easier plotting later
                self.rew_log[key].append(float(value.item()))
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    # ------------------------------------------------------------
    # Plot state signals (original)
    # ------------------------------------------------------------
    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 4   # one extra row for base position graphs
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(12, 10))
        fig.suptitle("Logger – Robot States", fontsize=14)
        log = self.state_log

        # create time axis from first logged signal
        for key, value in log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break

        # --- Base velocities ---
        a = axs[0, 0]
        if log["base_vel_x"]:
            a.plot(time, log["base_vel_x"], label="measured")
        if log["command_x"]:
            a.plot(time, log["command_x"], label="commanded")
        if log["est_lin_vel_x"]:
            a.plot(time, log["est_lin_vel_x"], label="est")
        a.set(xlabel="time [s]", ylabel="lin vel [m/s]", title="Base vel x")
        a.legend()

        a = axs[0, 1]
        if log["base_vel_y"]:
            a.plot(time, log["base_vel_y"], label="measured")
        if log["command_y"]:
            a.plot(time, log["command_y"], label="commanded")
        if log["est_lin_vel_y"]:
            a.plot(time, log["est_lin_vel_y"], label="est")
        a.set(xlabel="time [s]", ylabel="lin vel [m/s]", title="Base vel y")
        a.legend()

        a = axs[0, 2]
        if log["base_vel_yaw"]:
            a.plot(time, log["base_vel_yaw"], label="measured")
        if log["command_yaw"]:
            a.plot(time, log["command_yaw"], label="commanded")
        a.set(xlabel="time [s]", ylabel="ang vel [rad/s]", title="Base vel yaw")
        a.legend()

        # --- DOF position / velocity / z velocity ---
        a = axs[1, 0]
        if log["dof_pos"]:
            a.plot(time, log["dof_pos"], label="measured")
        if log["dof_pos_target"]:
            a.plot(time, log["dof_pos_target"], label="target")
        a.set(xlabel="time [s]", ylabel="pos [rad]", title="DOF Position")
        a.legend()

        a = axs[1, 1]
        if log["dof_vel"]:
            a.plot(time, log["dof_vel"], label="measured")
        if log["dof_vel_target"]:
            a.plot(time, log["dof_vel_target"], label="target")
        a.set(xlabel="time [s]", ylabel="vel [rad/s]", title="Joint Velocity")
        a.legend()

        a = axs[1, 2]
        if log["base_vel_z"]:
            a.plot(time, log["base_vel_z"], label="measured")
        a.set(xlabel="time [s]", ylabel="lin vel [m/s]", title="Base vel z")
        a.legend()

        # --- Contact forces ---
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f"foot {i}")
        a.set(xlabel="time [s]", ylabel="Force Z [N]", title="Contact forces Z")
        a.legend()

        # --- Power ---
        a = axs[2, 1]
        if log["power"]:
            a.plot(time, log["power"])
        a.set(xlabel="time [s]", ylabel="Power [W]", title="Total Power")
        a.legend()

        # --- Joint torques ---
        a = axs[2, 2]
        if log["dof_torque"]:
            a.plot(time, log["dof_torque"], label="measured")
        a.set(xlabel="time [s]", ylabel="Torque [Nm]", title="Torque")
        a.legend()

        # --- Base positions ---
        a = axs[3, 0]
        if "base_pos_x" in log and len(log["base_pos_x"]) > 0:
            a.plot(time, log["base_pos_x"], color="r")
            a.set(xlabel="time [s]", ylabel="pos [m]", title="Base Position X")

        a = axs[3, 1]
        if "base_pos_y" in log and len(log["base_pos_y"]) > 0:
            a.plot(time, log["base_pos_y"], color="g")
            a.set(xlabel="time [s]", ylabel="pos [m]", title="Base Position Y")

        a = axs[3, 2]
        if "base_pos_z" in log and len(log["base_pos_z"]) > 0:
            a.plot(time, log["base_pos_z"], color="b")
            a.set(xlabel="time [s]", ylabel="pos [m]", title="Base Position Z")

        # Hide empty panels
        for r in range(nb_rows):
            for c in range(nb_cols):
                if not axs[r, c].has_data():
                    axs[r, c].axis("off")

        plt.tight_layout()
        plt.show()

   
    # ------------------------------------------------------------
    # Stats print
    # ------------------------------------------------------------
    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / max(1, self.num_episodes)
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
        if self.plot_reward_process is not None:
            self.plot_reward_process.kill()