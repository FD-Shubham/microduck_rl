"""roller_sprint_2m's own push-capped rewards: past PUSH_REWARD_CMD_CAP the command factor of
every push-scaled reward must stop growing (measured on HF Jobs: microduck_mdp.wheel_speed_reward's
uncapped command factor made push training collapse once commands crossed the source recipe's 0.6
top -- see rewards.py's module docstring). No simulator: a stub env exposes only what each mdp
function under test reads."""

import re

import torch

from microduck_challenges.roller_sprint_2m import rewards
from microduck_challenges.roller_sprint_2m.env import PUSH_REWARD_CMD_CAP, make_env_cfg
from mjlab_microduck.tasks import mdp as microduck_mdp

_WHEELS = {"passive_LF_wheel": 0, "passive_LR_wheel": 1, "passive_RF_wheel": 2, "passive_RR_wheel": 3}


class _RobotData:
    def __init__(self, joint_vel=None, projected_gravity_b=None):
        self.joint_vel = joint_vel
        self.projected_gravity_b = projected_gravity_b


class _Robot:
    def __init__(self, data: _RobotData):
        self.data = data

    def find_joints(self, pattern):
        ids = [i for name, i in _WHEELS.items() if re.fullmatch(pattern, name)]
        assert ids, pattern
        return ids, None


class _SensorData:
    def __init__(self, current_air_time=None, current_contact_time=None):
        self.current_air_time = current_air_time
        self.current_contact_time = current_contact_time


class _Sensor:
    def __init__(self, data: _SensorData):
        self.data = data


class _CommandManager:
    def __init__(self, commands: torch.Tensor):
        self._commands = commands

    def get_command(self, name):
        return self._commands


class _Env:
    """A stand-in for ManagerBasedRlEnv exposing only command_manager and scene lookups."""

    def __init__(self, cmd_x, robot: _Robot, sensors: dict | None = None):
        cmd = torch.zeros(len(cmd_x), 3)
        cmd[:, 0] = torch.tensor(cmd_x, dtype=torch.float32)
        self.command_manager = _CommandManager(cmd)
        self._scene = {"robot": robot, **(sensors or {})}
        self.num_envs = len(cmd_x)

    @property
    def scene(self):
        return self._scene


def _wheel_robot(wheel_omega):
    """4 wheels all spinning at wheel_omega (one row per env)."""
    vel = torch.tensor([[w, w, w, w] for w in wheel_omega], dtype=torch.float32)
    return _Robot(_RobotData(joint_vel=vel))


# ── capped_wheel_speed_reward ────────────────────────────────────────────────


def test_matches_the_source_formula_below_the_cap():
    env = _Env(cmd_x=[0.2, 0.4], robot=_wheel_robot([20.0, 5.0]))
    got = rewards.capped_wheel_speed_reward(env, "twist", cmd_cap=0.6)
    want = microduck_mdp.wheel_speed_reward(env, "twist", bidirectional=False)
    assert torch.allclose(got, want)


def test_commands_at_and_above_the_cap_give_the_same_reward():
    env = _Env(cmd_x=[0.6, 1.0, 2.0], robot=_wheel_robot([20.0, 20.0, 20.0]))
    got = rewards.capped_wheel_speed_reward(env, "twist", cmd_cap=0.6)
    assert torch.allclose(got, got[0].expand_as(got))
    assert got[0].item() > 0.0


def test_non_positive_commands_give_zero():
    env = _Env(cmd_x=[0.0, -1.0], robot=_wheel_robot([20.0, 20.0]))
    got = rewards.capped_wheel_speed_reward(env, "twist", cmd_cap=0.6)
    assert torch.equal(got, torch.zeros(2))


def test_wheel_speed_term_uses_the_capped_reward_at_the_source_recipes_top():
    cfg = make_env_cfg()
    term = cfg.rewards["wheel_speed"]
    assert term.func is rewards.capped_wheel_speed_reward
    assert term.params["cmd_cap"] == PUSH_REWARD_CMD_CAP == 0.6


# ── the other three capped wrappers: cap-invariance + agreement with the source below the cap ──


def test_capped_skating_air_time_matches_source_below_cap_and_caps_above_it():
    sensors = {"feet": _Sensor(_SensorData(current_air_time=torch.tensor([[0.2, 0.2]])))}
    below = _Env(cmd_x=[0.3], robot=_Robot(_RobotData()), sensors=sensors)
    got = rewards.capped_skating_air_time_reward(below, "feet", "twist", cmd_cap=0.6)
    want = microduck_mdp.skating_air_time_reward(below, "feet", "twist")
    assert torch.allclose(got, want)

    at_cap = _Env(cmd_x=[0.6], robot=_Robot(_RobotData()), sensors=sensors)
    above_cap = _Env(cmd_x=[2.0], robot=_Robot(_RobotData()), sensors=sensors)
    r_cap = rewards.capped_skating_air_time_reward(at_cap, "feet", "twist", cmd_cap=0.6)
    r_above = rewards.capped_skating_air_time_reward(above_cap, "feet", "twist", cmd_cap=0.6)
    assert torch.allclose(r_cap, r_above)
    assert r_cap.item() > 0.0


def test_capped_single_support_matches_source_below_cap_and_caps_above_it():
    sensors = {"feet": _Sensor(_SensorData(current_contact_time=torch.tensor([[0.1, 0.0]])))}
    below = _Env(cmd_x=[0.3], robot=_Robot(_RobotData()), sensors=sensors)
    got = rewards.capped_single_support_reward(below, "feet", "twist", cmd_cap=0.6)
    want = microduck_mdp.single_support_reward(below, "feet", "twist")
    assert torch.allclose(got, want)

    at_cap = _Env(cmd_x=[0.6], robot=_Robot(_RobotData()), sensors=sensors)
    above_cap = _Env(cmd_x=[2.0], robot=_Robot(_RobotData()), sensors=sensors)
    r_cap = rewards.capped_single_support_reward(at_cap, "feet", "twist", cmd_cap=0.6)
    r_above = rewards.capped_single_support_reward(above_cap, "feet", "twist", cmd_cap=0.6)
    assert torch.allclose(r_cap, r_above)
    assert r_cap.item() > 0.0


def test_capped_forward_lean_matches_source_below_cap_and_caps_above_it():
    def robot_with_lean():
        return _Robot(_RobotData(projected_gravity_b=torch.tensor([[0.262, 0.0, -0.96]])))

    below = _Env(cmd_x=[0.3], robot=robot_with_lean())
    got = rewards.capped_forward_lean_reward(below, "twist", cmd_cap=0.6, target_pitch=0.262, std=0.1)
    want = microduck_mdp.forward_lean_reward(below, "twist", target_pitch=0.262, std=0.1)
    assert torch.allclose(got, want)

    at_cap = _Env(cmd_x=[0.6], robot=robot_with_lean())
    above_cap = _Env(cmd_x=[2.0], robot=robot_with_lean())
    r_cap = rewards.capped_forward_lean_reward(at_cap, "twist", cmd_cap=0.6, target_pitch=0.262, std=0.1)
    r_above = rewards.capped_forward_lean_reward(
        above_cap, "twist", cmd_cap=0.6, target_pitch=0.262, std=0.1
    )
    assert torch.allclose(r_cap, r_above)
    assert r_cap.item() > 0.0
