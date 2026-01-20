# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import (
    get_args,
    export_policy_as_jit,
    export_mlp_as_onnx,
    task_registry,
    Logger,
)

import numpy as np
import torch
import matplotlib.pyplot as plt


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 10
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)

    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    env_cfg.terrain.max_init_terrain_level = 4
    # env_cfg.terrain.curriculum = True
    # env_cfg.noise.add_noise = True
    # env_cfg.noise.noise_level = 0.5
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_restitution = False
    # env_cfg.domain_rand.randomize_base_com = False
    # env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.push_interval_s = 3
    # env_cfg.domain_rand.randomize_Kp = False
    # env_cfg.domain_rand.randomize_Kd = False
    # env_cfg.domain_rand.randomize_motor_torque = False
    # env_cfg.domain_rand.randomize_default_dof_pos = False
    # env_cfg.domain_rand.randomize_action_delay = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # ✅  Force an explicit full reset before starting the main loop
    print("🔄 Performing explicit full reset before play loop...")
    env.reset()
    # get robot_type
    robot_type = os.getenv("ROBOT_TYPE")
    # Default commands per robot type / task
    if robot_type.startswith("PF"):
        # PF: 4 commands (example from README)
        commands_val = to_torch([0.5, 0.0, 0.0, 0.0], device=env.device)
    elif robot_type == "WF_TRON1A":
        if args.task == "wheelfoot_stand":
            # WF stand: 4 commands -> [lin_x, lin_y, yaw, height_target]
            # If height_target CLI arg is provided, use it; otherwise use middle of config range
            if hasattr(args, "height_target") and args.height_target is not None:
                height_target = args.height_target
            else:
                h_min, h_max = env_cfg.commands.ranges.height_target
                height_target = 0.5 * (h_min + h_max)
            commands_val = to_torch([0.0, 0.0, 0.0, height_target], device=env.device)
        else:
            # Other WF tasks: 3 commands [lin_x, lin_y, yaw]
            commands_val = to_torch([0.0, 0.0, 0.0], device=env.device)
    else:
        # SF or other: 5 commands as originally
        commands_val = to_torch([1.5, 0.0, 0.0, 0.0, 0.0], device=env.device)
    action_scale = env.cfg.control.action_scale_pos if robot_type == "WF_TRON1A"\
        else env.cfg.control.action_scale
    obs, obs_history, commands, _ = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    # train_cfg.runner.checkpoint = -1

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            args.task,
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor,
            path,
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            path,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging / manual command control
    joint_index = 1  # which joint is used for logging
    stop_state_log = 500  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1.0, 1.0, 0.0])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    est = None
    idle_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    prev_actions = torch.zeros_like(env.actions, device=env.device)

    for i in range(10 * int(env.max_episode_length)):
        est = encoder(obs_history)
        actions = policy(torch.cat((est, obs, commands), dim=-1).detach())

        # Set commands
        if robot_type == "WF_TRON1A" and args.task == "wheelfoot_stand":
            # Only override the monitored environment's commands so we can control its height target
            env.commands[robot_index, : commands_val.shape[0]] = commands_val
        else:
            # For other tasks, use the same commands for all envs
            env.commands[:, : commands_val.shape[0]] = commands_val

        obs, rews, dones, infos, obs_history, commands, _ = env.step(
            actions.detach()
        )


        

        # --- Custom termination: policy idling behaviour ---
        # Detect if actions haven’t changed much
        idle_actions = torch.all(torch.isclose(actions, prev_actions, atol=1e-3), dim=1)
        # Count consecutive idle steps
        idle_steps = torch.where(idle_actions, idle_steps + 1, torch.zeros_like(idle_steps))
        # Keep copy for next step
        prev_actions = actions.clone()

        # Detect joints near default positions
        is_default = torch.all(
            torch.isclose(env.dof_pos, env.raw_default_dof_pos, atol=0.02),
            dim=1
        )

        # Combine both conditions: idle for > 50 steps (≈ 0.5 s) and at default
        stop_condition = (idle_steps > 50) & is_default

        if torch.any(stop_condition):
            stopped_envs = torch.nonzero(stop_condition).flatten()
            print(f"🧭 Policy‑idle reset for envs {stopped_envs.tolist()}", flush=True)
            env.reset_idx(stopped_envs)
            idle_steps[stopped_envs] = 0        # reset counters
            prev_actions[stopped_envs] = 0




        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            target_position[2] = 0
            camera_position = target_position + camera_offset
            # env.set_camera(camera_position, target_position)

        # --- Log uniquement la simulation du robot d’intérêt ---
        if not hasattr(env, "first_reset_done"):
            env.first_reset_done = False
            env.first_reset_detected = False

        # Vérifie uniquement le reset du robot_index ciblé
        if not env.first_reset_done and env.reset_buf[robot_index]:
            env.first_reset_done = True
            env.first_reset_detected = True
            episode_time_s = env.episode_length_buf[robot_index].item() * env.dt
            print(f"✅ Premier reset détecté pour l'env {robot_index} "
                f"(t = {episode_time_s:.2f}s) — arrêt de l’enregistrement.", flush=True)

        # On n’enregistre que si on est avant le premier reset
        if not env.first_reset_done:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item() * action_scale,
                    "dof_pos": (
                        env.dof_pos[robot_index, joint_index]
                        - env.raw_default_dof_pos[joint_index]
                    ).item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "base_pos_x": env.base_position[robot_index, 0].item(),
                    "base_pos_y": env.base_position[robot_index, 1].item(),
                    "base_pos_z": env.base_position[robot_index, 2].item(),
                    "power": torch.sum(env.power[robot_index, :]).item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
            if est is not None:
                logger.log_states(
                    {
                        "est_lin_vel_x": est[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": est[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": est[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                    }
                )

        # Quand la première simulation s’est terminée, on peut tracer une fois
        if getattr(env, "first_reset_detected", False):
            logger.plot_states()
            env.first_reset_detected = False  # pour éviter de re‑tracer à chaque itération

        # if i < stop_state_log:
        #     logger.log_states(
        #         {
        #             "dof_pos_target": actions[robot_index, joint_index].item() * action_scale,
        #             "dof_pos": (
        #                 env.dof_pos[robot_index, joint_index]
        #                 - env.raw_default_dof_pos[joint_index]
        #             ).item(),
        #             "dof_vel": env.dof_vel[robot_index, joint_index].item(),
        #             "dof_torque": env.torques[robot_index, joint_index].item(),
        #             "command_x": env.commands[robot_index, 0].item(),
        #             "command_y": env.commands[robot_index, 1].item(),
        #             "command_yaw": env.commands[robot_index, 2].item(),
        #             "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
        #             "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
        #             "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
        #             "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
        #             "base_pos_x": env.base_position[robot_index, 0].item(),
        #             "base_pos_y": env.base_position[robot_index, 1].item(),
        #             "base_pos_z": env.base_position[robot_index, 2].item(),
        #             "power": torch.sum(env.power[robot_index, :]).item(),
        #             "contact_forces_z": env.contact_forces[
        #                 robot_index, env.feet_indices, 2
        #             ]
        #             .cpu()
        #             .numpy(),
        #         }
        #     )
        #     # print(torch.sum(env.power[robot_index, :]).item())
        #     if est != None:
        #         logger.log_states(
        #             {
        #                 "est_lin_vel_x": est[robot_index, 0].item()
        #                 / env.cfg.normalization.obs_scales.lin_vel,
        #                 "est_lin_vel_y": est[robot_index, 1].item()
        #                 / env.cfg.normalization.obs_scales.lin_vel,
        #                 "est_lin_vel_z": est[robot_index, 2].item()
        #                 / env.cfg.normalization.obs_scales.lin_vel,
        #             }
        #         )
        # elif i == stop_state_log:
        #     logger.plot_states()

        # if 0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes > 0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i == stop_rew_log:
        #     logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
