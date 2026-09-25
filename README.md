# Microduck challenges

One folder per challenge open on the Microduck Arena. Each is a complete,
working training environment: train it unchanged and the robot competes.

| Folder | Arena event | Task |
|---|---|---|
| (filled in as challenges are added) | | |

## Train from a shell

Needs Python 3.12 and `uv`. Training needs an NVIDIA GPU, or Hugging Face Jobs:

```bash
git clone https://github.com/pollen-robotics/microduck-challenges && cd microduck-challenges
export HF_TOKEN=$(cat ~/.cache/huggingface/token)   # while the Arena wheel is private
uv sync --group workshop
uv run train Mjlab-Sprint2m-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3000 \
    --agent.seed 1 --agent.logger tensorboard --agent.run-name first
# no GPU: add --hf-jobs --no-wandb
```

A challenge's knobs are in its `challenge.toml` (`[params]`); its whole
environment is in `env.py`. Edit either, commit, train.

Then `uv run publish --run logs/rsl_rl/<folder>/<run> --repo <you>/<name>` and
paste the repo on the challenge's page on the Arena.
