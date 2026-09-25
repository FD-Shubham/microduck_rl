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
}

ROOT = Path(microduck_challenges.__file__).parent


def _contract(folder):
    return challenge.load(ROOT / folder / "tasks.py")


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
