"""roller_sprint_2m's own reward functions (README: "to add a custom reward, write the
function in a rewards.py next to the challenge's env.py").

Every push-scaled term below caps its command factor at PUSH_REWARD_CMD_CAP (env.py):
measured on HF Jobs (4096 envs, seed 1), letting the command factor grow past the source
recipe's top command (0.6) made training collapse. `microduck_mdp.wheel_speed_reward`'s
command factor is `clamp(cmd_x, min=0)`, unbounded above: at weight 10 the reward is worth
up to 6/step at cmd_x 0.6 (the source recipe's top, known to train) but up to 20/step at
the Arena's 2.0 -- ten times `upright` (weight 2.0) -- so past 0.6 the policy trades balance
for spin and falls (run 1: collapsed from iteration 250 commanding up to 2.00 directly; run 2:
stable through iteration 1002 on a curriculum topping out at 1.0, then collapsed at iteration
1180 right after that stage, episode length 678 -> 72, fell_over 3 -> 47).

Each function below is a plain copy of its `microduck_mdp` counterpart (non-bidirectional path,
wheel joint lookup included) with its command factor changed from `clamp(cmd_x, min=0)` to
`clamp(cmd_x, min=0, max=cmd_cap)` -- no other change, no calls into microduck_mdp's private
helpers.
"""

import torch

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


def capped_wheel_speed_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    cmd_cap: float,
    wheel_radius: float = 0.0175,
    vel_scale: float = 0.5,
) -> torch.Tensor:
    """`microduck_mdp.wheel_speed_reward` (bidirectional=False), command factor capped at
    cmd_cap: reward wheel spin proportional to commanded push, but push credit stops growing
    past cmd_cap. Silent (0) for cmd_x <= 0 -- braking is handled by the braking reward."""
    cmd_x = env.command_manager.get_command(command_name)[:, 0]  # (B,)

    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    forward_omega = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]] + vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 4.0

    omega_scale = vel_scale / wheel_radius
    push = torch.clamp(cmd_x, min=0.0, max=cmd_cap)
    return push * torch.tanh(torch.clamp(forward_omega, min=0.0) / omega_scale)


def _forward_progress_gate(env: ManagerBasedRlEnv, v_ref: float) -> torch.Tensor | None:
    """Copy of microduck_mdp's private `_forward_progress_gate` (0->1 ramp in forward speed,
    None when disabled): shared by the two capped wrappers below that gate on forward progress,
    without reaching into microduck_mdp's own private helper."""
    if v_ref <= 0.0:
        return None
    v_fwd = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return (v_fwd.clamp(min=0.0) / v_ref).clamp(max=1.0)


def capped_skating_air_time_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    cmd_cap: float,
    threshold_min: float = 0.05,
    threshold_max: float = 0.4,
    vel_gate_ref: float = 0.0,
) -> torch.Tensor:
    """`microduck_mdp.skating_air_time_reward`, command factor capped at cmd_cap: its magnitude
    (feet-in-range count times cmd_x) grows unbounded with cmd_x, same failure as wheel_speed."""
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    reward = reward * torch.clamp(cmd_x, min=0.0, max=cmd_cap)
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        reward = reward * gate
    return reward


def capped_single_support_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    cmd_cap: float,
    vel_gate_ref: float = 0.0,
    double_penalty: float = 0.25,
) -> torch.Tensor:
    """`microduck_mdp.single_support_reward`, command factor capped at cmd_cap: both the
    single-support reward and the double-support penalty scale with cmd_x unbounded, same
    failure as wheel_speed."""
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None

    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)  # (num_envs,)
    single = (n_contact == 1).float()
    double = (n_contact >= 2).float()

    cmd_x = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.0, max=cmd_cap)
    single_r = single * cmd_x
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        single_r = single_r * gate
    return single_r - double_penalty * double * cmd_x


def capped_forward_lean_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    cmd_cap: float,
    target_pitch: float = 0.08,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",)),
) -> torch.Tensor:
    """`microduck_mdp.forward_lean_reward`, command factor capped at cmd_cap: the push factor
    that gates the lean-shaping gaussian grows with cmd_x unbounded, same failure as
    wheel_speed."""
    asset: Entity = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    forward_lean = asset.data.projected_gravity_b[:, 0]
    push = torch.clamp(cmd_x, min=0.0, max=cmd_cap)
    return push * torch.exp(-((forward_lean - target_pitch) ** 2) / (std ** 2))
