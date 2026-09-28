# Microduck challenges

One folder per challenge open on the Microduck Arena. Each is a complete,
working training environment: train it unchanged and the robot competes.

| Folder | Arena event | Task |
|---|---|---|
| sprint_2m | sprint-2m | Mjlab-Sprint2m-MicroDuck |
| swag_contest | swag-contest | Mjlab-SwagContest-MicroDuck |
| hill_climb_1m5 | hill-climb-1m5 | Mjlab-HillClimb1m5-MicroDuck |
| roller_sprint_2m | roller-sprint-2m | Mjlab-RollerSprint2m-MicroDuck |

## The workshop

Each challenge's page on the Arena has a **Train my robot** button with one line to copy:

```bash
uv run https://pollen-robotics-microduck-arena.hf.space/workshop.py sprint-2m
```

It works in the clone you run it in (or clones this repo next to you) and opens
the workshop in your browser: Preflight, Train (on your NVIDIA GPU, or on
Hugging Face Jobs without one), Test (the Arena's own evaluator, on your
machine), Publish (to your Hugging Face account), then back to the Arena to
enter it. From a clone, `uv run --group workshop workshop` lists the challenges
and `uv run --group workshop workshop sprint-2m` opens one.

## Train from a shell

Needs Python 3.12 and `uv`. Training needs an NVIDIA GPU, or Hugging Face Jobs.

While the Arena wheel sits on a private Hub repo, EVERY `uv` command in this
checkout needs `HF_TOKEN` — even a plain `uv run` re-checks the lock and
re-fetches the wheel URL, and gets a 401 without it. Log in once
(`hf auth login`), then export the token in the shell (or its profile) before
any `uv` command:

```bash
git clone https://github.com/pollen-robotics/microduck-challenges && cd microduck-challenges
export HF_TOKEN=$(cat ~/.cache/huggingface/token)   # while the Arena wheel is private
uv sync --group workshop
uv run train Mjlab-Sprint2m-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3000 \
    --agent.seed 1 --agent.logger tensorboard --agent.run-name first
# no GPU: add --hf-jobs --no-wandb
```

A challenge's knobs are in its `challenge.toml` (`[params]`); its whole
environment is in `env.py`. Edit either, commit, train. To add a custom
reward, write the function in a `rewards.py` next to the challenge's `env.py`
and add it to `cfg.rewards` in `env.py`.

`--hf-jobs` trains on Hugging Face's infrastructure instead of a local GPU;
nothing lands in `logs/rsl_rl/` here. Its checkpoints go to the Hub repo the
submission prints (`[ckpt] checkpoints -> <repo>`), mirroring `logs/rsl_rl/` —
bring them back before publishing:

```bash
uv run hf download <that repo> --local-dir logs/rsl_rl
```

Then `uv run publish --run logs/rsl_rl/<folder>/<run> --repo <you>/<name> --no-private`
and paste the repo on the challenge's page on the Arena (`publish` creates the
Hub repo private by default; the Arena reads with its own token, so it needs
`--no-private` to see it).

## Tests

```bash
uv run --group workshop pytest
```

(the tests import the Arena, which lives in the `workshop` group).
