"""Pin the Arena's vendored BAM XL330 servo (arena.actuator) to bam's own
numbers and math, so the two can never silently drift apart.

This is the only environment with arena, bam and torch installed together.
Until the Arena's 0.3.0 wheel is published, run against the arena-bam
worktree explicitly:

    uv run --no-config --frozen --group workshop \\
        --with-editable ~/Projects/HuggingFace/microduck-arena/.claude/worktrees/arena-bam \\
        pytest tests/test_arena_actuator_parity.py -q

If a vendored number or formula here diverges from bam, the fix belongs in
the Arena (arena/actuator.py), not in this test.
"""

import numpy as np
import pytest
import torch
from bam.mjlab import BamActuator
from bam.model import load_model
from mjlab_microduck.robot import microduck_constants

import arena.actuator as A

# bam Model parameter name -> arena.actuator constant, for the m6-fitted
# XL330 parameters (bam/params/xl330/m6.json, loaded via bam.model.load_model).
_MODEL_PARAMS = {
    "kt": "KT",
    "R": "R",
    "armature": "ARMATURE",
    "friction_base": "FRICTION_BASE",
    "friction_stribeck": "FRICTION_STRIBECK",
    "load_friction_motor": "LOAD_MOTOR",
    "load_friction_external": "LOAD_EXTERNAL",
    "load_friction_motor_stribeck": "LOAD_MOTOR_STRIBECK",
    "load_friction_external_stribeck": "LOAD_EXTERNAL_STRIBECK",
    "load_friction_motor_quad": "LOAD_MOTOR_QUAD",
    "load_friction_external_quad": "LOAD_EXTERNAL_QUAD",
    "dtheta_stribeck": "DTHETA_STRIBECK",
    "alpha": "ALPHA",
    "friction_viscous": "FRICTION_VISCOUS",
}


@pytest.fixture(scope="module")
def bam_model():
    return load_model(motor_name="xl330", model="m6")


def test_model_parameters_match_bam(bam_model):
    for bam_name, arena_name in _MODEL_PARAMS.items():
        bam_value = getattr(bam_model, bam_name).value
        arena_value = getattr(A, arena_name)
        assert arena_value == bam_value, (
            f"arena.actuator.{arena_name} = {arena_value!r}, "
            f"bam's {bam_name} = {bam_value!r}"
        )


def test_firmware_constants_match_bams_xl330_actuator(bam_model):
    # bam/dynamixel/actuator.py: fixed firmware constants (not fitted), on the
    # XL330Actuator instance bam.model.load_model attaches.
    act = bam_model.actuator
    assert A.ERROR_GAIN == act.error_gain
    assert A.MAX_PWM == act.max_pwm
    assert A.MAX_CURRENT == act.max_current


def test_friction_constraint_stiffness_matches_bam_mjlab():
    # bam/mjlab.py: BamActuator's solref/solimp for the joint-friction
    # constraint (stiff_frictionloss=True, training's default).
    assert A.SOLREF_FRICTION == BamActuator._STIFF_SOLREF_FRICTION
    assert A.SOLIMP_FRICTION == BamActuator._STIFF_SOLIMP_FRICTION


def test_kp_fw_matches_training():
    assert A.KP_FW == microduck_constants.actuators.kp_fw == 200.0


def test_nominal_values_sit_inside_trainings_randomisation_ranges():
    cfg = microduck_constants.actuators

    vin_lo, vin_hi = cfg.vin_range
    assert vin_lo <= A.VIN_NOMINAL <= vin_hi
    assert A.VIN_NOMINAL == (vin_lo + vin_hi) / 2

    drop_lo, drop_hi = cfg.vin_drop_gain_range
    assert drop_lo <= A.VIN_DROP_GAIN <= drop_hi
    assert A.VIN_DROP_GAIN == (drop_lo + drop_hi) / 2

    assert A.VIN_MIN == cfg.vin_min
    assert A.VIN_MAX == max(vin_lo, vin_hi)
    assert A.DELAY_STEPS == (cfg.delay_min_lag, cfg.delay_max_lag)


def test_force_limit_matches_bam_mjlabs_forcerange_formula(bam_model):
    # bam/mjlab.py edit_spec: force_limit = max(vin_range) * kt / R.
    expected = A.VIN_MAX * bam_model.kt.value / bam_model.R.value
    assert A.FORCE_LIMIT == expected


def test_friction_budget_matches_bam_mjlab_across_20000_random_states(bam_model):
    rng = np.random.default_rng(0)
    n = 20_000
    mot = rng.uniform(-1.2, 1.2, n)
    ext = rng.uniform(-1.2, 1.2, n)
    dq = rng.uniform(-8, 8, n) * (rng.random(n) < 0.5) + rng.normal(0, 0.5, n)

    # bam.mjlab.BamActuator._compute_friction_budget (training's torch path),
    # instantiated without going through mjlab's full actuator setup.
    ref_actuator = BamActuator.__new__(BamActuator)
    ref_actuator._bam_model = bam_model
    stribeck = np.exp(
        -np.power(np.abs(dq) / bam_model.dtheta_stribeck.value, bam_model.alpha.value)
    )
    reference = ref_actuator._compute_friction_budget(
        torch.tensor(mot), torch.tensor(ext), torch.tensor(stribeck)
    ).numpy()

    vendored = A.friction_budget(mot, ext, dq)

    max_diff = np.abs(vendored - reference).max()
    assert max_diff < 1e-9


def test_control_torque_matches_bam_actuator_across_2000_random_states(bam_model):
    rng = np.random.default_rng(1)
    n = 2_000
    qerr = rng.uniform(-0.6, 0.6, n)
    dq = rng.uniform(-8, 8, n) * (rng.random(n) < 0.5) + rng.normal(0, 0.5, n)
    vin = rng.uniform(6.0, 8.2, n)

    # bam.actuator's numpy backend: compute_control (firmware position loop +
    # current window) then compute_torque (DC-motor equation), one sample at
    # a time as bam.mujoco.MujocoController drives it.
    act = bam_model.actuator
    act.kp = A.KP_FW  # training's firmware stiffness (bam's XL330 default is 400)
    reference = np.empty(n)
    for i in range(n):
        act.vin = vin[i]
        control = act.compute_control(qerr[i], 0.0, dq[i], 0.005)
        reference[i] = act.compute_torque(control, True, 0.0, dq[i])

    vendored = A.control_torque(qerr, np.zeros(n), dq, vin)

    max_diff = np.abs(vendored - reference).max()
    assert max_diff < 1e-9
