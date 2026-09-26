"""Every challenge folder: it registers, it trains for an open event, its knobs are live."""

import importlib
import math
from pathlib import Path

import pytest
from mjlab.tasks.registry import list_tasks

import microduck_challenges  # noqa: F401  (registers every task)
from arena.manifest import load_event
from mjlab_microduck import challenge
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_WALK_ROBOT_CFG,
    MICRODUCK_WALK_ROLLERS_ROBOT_CFG,
)

# folder, and for each exposed knob: (reward term whose weight it sets, value to try).
CHALLENGES = {
    "sprint_2m": {
        "forward_reward": ("track_linear_velocity", 3.7),
        "upright_reward": ("upright", 1.3),
        "step_reward": ("air_time", 4.1),
    },
    "swag_contest": {
        "step_reward": ("air_time", 5.2),
        "upright_reward": ("upright", 0.9),
        "head_reward": ("head_pose_tracking", 2.6),
    },
    "hill_climb_1m5": {
        "forward_reward": ("track_linear_velocity", 3.7),
        "upright_reward": ("upright", 1.3),
        "step_reward": ("air_time", 4.1),
        "max_slope_deg": (None, 12.0),
    },
    "roller_sprint_2m": {
        "speed_reward": ("wheel_speed", 13.5),
        "upright_reward": ("upright", 1.3),
    },
}

ROOT = Path(microduck_challenges.__file__).parent


def _contract(folder):
    return challenge.load(ROOT / folder / "tasks.py")


# The challenges whose Arena event races a commanded speed; swag is judged on style.
RACES = ["sprint_2m", "hill_climb_1m5", "roller_sprint_2m"]


@pytest.mark.parametrize("folder", RACES)
def test_training_covers_the_speeds_the_arena_commands(folder):
    env = importlib.import_module(f"microduck_challenges.{folder}.env")
    cfg = env.make_env_cfg()
    lin_vel_x = cfg.commands["twist"].ranges.lin_vel_x
    event = load_event(_contract(folder).event)
    assert lin_vel_x[0] <= event.sweep.lo
    ranges = cfg.commands["twist"].ranges
    assert ranges.lin_vel_y == (0.0, 0.0) and ranges.ang_vel_z == (0.0, 0.0)


@pytest.mark.parametrize("folder", ["sprint_2m", "hill_climb_1m5"])
def test_walking_training_reaches_the_top_of_the_sweep(folder):
    # roller_sprint_2m is excluded: its top command is the robot's measured push
    # limit, not the Arena's sweep top (see roller_sprint_2m/env.py).
    env = importlib.import_module(f"microduck_challenges.{folder}.env")
    cfg = env.make_env_cfg()
    lin_vel_x = cfg.commands["twist"].ranges.lin_vel_x
    event = load_event(_contract(folder).event)
    assert lin_vel_x[1] >= event.sweep.hi


def test_the_roller_trains_with_the_arenas_free_wheels():
    # The Arena's roller model's wheel bearings are frictionless (frictionloss=0.0
    # in robot_allcollisions_rollers.xml's passive_joint default); a policy trained
    # against added bearing friction breaks once raced there (see roller_sprint_2m/env.py).
    env = importlib.import_module("microduck_challenges.roller_sprint_2m.env")
    cfg = env.make_env_cfg()
    assert "randomize_wheel_friction" not in cfg.events
    assert "wheel_friction" not in cfg.curriculum


@pytest.mark.parametrize("folder", CHALLENGES)
def test_the_task_is_registered(folder):
    assert _contract(folder).task in list_tasks()


@pytest.mark.parametrize("folder", CHALLENGES)
def test_the_event_is_open_on_the_installed_arena(folder):
    event = load_event(_contract(folder).event)
    assert event.status == "open", f"{folder} trains for {event.id}, which is {event.status}"


@pytest.mark.parametrize("folder", CHALLENGES)
def test_the_robot_matches_the_events_rig(folder):
    env = importlib.import_module(f"microduck_challenges.{folder}.env")
    robot = env.make_env_cfg().scene.entities["robot"]
    expected = {"legs": MICRODUCK_WALK_ROBOT_CFG, "rollers": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    assert robot is expected[load_event(_contract(folder).event).rig]


@pytest.mark.parametrize("folder", CHALLENGES)
def test_runs_log_under_the_folder_name(folder):
    env = importlib.import_module(f"microduck_challenges.{folder}.env")
    assert env.RL_CFG.experiment_name == folder


def test_every_toml_knob_is_in_the_table():
    """A knob added to challenge.toml but not to CHALLENGES would otherwise go unchecked."""
    for folder in CHALLENGES:
        assert set(CHALLENGES[folder]) == set(challenge.params(ROOT / folder / "tasks.py")), folder


def test_every_challenge_folder_has_a_row():
    """A folder without a CHALLENGES row goes unchecked by every test above."""
    assert set(CHALLENGES) == {p.parent.name for p in ROOT.glob("*/challenge.toml")}


@pytest.mark.parametrize("folder", CHALLENGES)
def test_knobs_reach_the_config(folder, tmp_path, monkeypatch):
    """Rewrite the knobs to unusual values; the rebuilt config must carry them."""
    toml = ROOT / folder / "challenge.toml"
    text = toml.read_text()
    for knob, (_, value) in CHALLENGES[folder].items():
        text = _set_value(text, knob, value)
    copy = tmp_path / "challenge.toml"
    copy.write_text(text)
    real_params = challenge.params
    monkeypatch.setattr(challenge, "params", lambda f: real_params(copy))

    env = importlib.import_module(f"microduck_challenges.{folder}.env")
    cfg = env.make_env_cfg()
    for knob, (term, value) in CHALLENGES[folder].items():
        if term is None:
            continue
        assert math.isclose(cfg.rewards[term].weight, value), f"{knob} does not set {term}"


def _set_value(text: str, knob: str, value: float) -> str:
    head = f"[params.{knob}]"
    assert head in text, f"no {head}"
    before, after = text.split(head, 1)
    lines = after.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.split("=")[0].strip() == "value":
            lines[i] = f"value = {value}\n"
            return before + head + "".join(lines)
    raise AssertionError(f"{head} has no value")


def test_the_hill_trains_on_slopes_up_to_the_knob(tmp_path, monkeypatch):
    toml = ROOT / "hill_climb_1m5" / "challenge.toml"
    # Read the SHIPPED value before challenge.params gets monkeypatched below.
    shipped_max_slope_deg = challenge.params(ROOT / "hill_climb_1m5" / "tasks.py")["max_slope_deg"]
    copy = tmp_path / "challenge.toml"
    copy.write_text(_set_value(toml.read_text(), "max_slope_deg", 12.0))
    real_params = challenge.params
    monkeypatch.setattr(challenge, "params", lambda f: real_params(copy))
    from microduck_challenges.hill_climb_1m5.env import make_env_cfg

    terrain = make_env_cfg().scene.terrain
    assert terrain.terrain_type == "generator"
    slope = terrain.terrain_generator.sub_terrains["slope"]
    assert math.isclose(slope.slope_range[1], math.tan(math.radians(12.0)))
    assert shipped_max_slope_deg > 8.0, "must cover the course's 8°"
