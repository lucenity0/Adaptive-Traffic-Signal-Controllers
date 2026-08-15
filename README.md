# Adaptive Traffic Signal Controllers

A PPO agent that decides when a junction changes phase — trained under a
traffic curriculum, constrained so emergency vehicles can preempt it, and
instrumented so you can ask *why* it chose a phase rather than taking the
answer on faith.

It runs in two places: a fast built-in simulator for training and ablations,
and a **SUMO** junction over **TraCI** for the real thing.

---

## Why

Signal controllers that learn tend to be opaque, and opaque is a hard sell for
infrastructure — a junction that cannot explain itself is a junction nobody
signs off. Two things here address that:

- **Constraints the policy cannot argue with.** Minimum and maximum green
  times, and an emergency preemption override, are enforced by the
  environment rather than learned as preferences. The agent optimises inside
  the envelope; it does not get to leave it.
- **Attribution per decision.** Integrated gradients against the chosen action
  logit, so any phase change comes with the state features that drove it.

---

## The two environments

### `MixedTrafficSignalEnv` — the built-in simulator

A queueing model of a single junction: arrivals, service, and phases, with
emergency vehicles appearing at a configurable rate. It exists so training and
ablations run in seconds without a SUMO install, and it is what the curriculum
and baselines use.

Per step it applies the phase action, serves the current phase, spawns
emergencies, and scores the result.

**Reward** — queue length and accumulated waiting are penalised, switching
costs a little (so the policy cannot flicker between phases), delaying an
emergency costs more, clearing one pays, and the total is clipped:

| Term | Default |
|---|---|
| `queue_penalty_coef` | 1.0 |
| `wait_penalty_coef` | 0.10 |
| `switch_penalty` | 0.15 |
| `emergency_delay_penalty` | 0.35 |
| `emergency_served_bonus` | 2.0 |
| `reward_clip` | ±5.0 |

**Constraints**

| Setting | Default |
|---|---|
| `min_green_steps` | 8 |
| `max_green_steps` | 45 |
| `queue_cap` | 60 |
| `emergency_prob` | 0.02 |
| `preemption_cap_steps` | 12 |
| `preemption_recovery_steps` | 8 |

`_preemption_policy_override` is where the agent's action is overruled when an
emergency is active, capped so preemption cannot be held indefinitely, with a
recovery window afterwards.

### `SumoTraciAdapterEnv` — a real junction

Talks to SUMO over TraCI in a bidirectional loop: read lane queues, decide,
set the phase, step the simulation, read again.

It discovers what to control rather than being told
(`_discover_control_target`), so it can be pointed at an arbitrary
single-junction scenario. `available()` reports whether TraCI and SUMO are
importable, so the rest of the code can degrade to the built-in simulator
instead of failing at import time.

The junction in [`scenario/`](scenario) is a ready-made one to start from.

---

## The agent

`PPOAgent` — a standard clipped-objective PPO over a small actor/critic pair
(two hidden layers, 64 units, tanh).

| Hyperparameter | Default |
|---|---|
| `episodes` | 160 |
| `steps_per_episode` | 240 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_ratio` | 0.2 |
| `actor_lr` / `critic_lr` | 3e-4 / 1e-3 |
| `update_epochs` | 4 |
| `entropy_coef` | 0.01 |
| `value_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `hidden_size` | 64 |
| `seed` | 42 |

Advantages come from GAE; updates run several epochs over each trajectory with
gradient clipping.

### Curriculum

`curriculum_schedule` ramps two things across training — the share of
autonomous vehicles and overall difficulty — in one of three modes:

- `linear` (default) — ramp both across the run
- `max` — train at full difficulty throughout
- `off` — hold at the starting conditions

The point is that a policy meeting congestion for the first time in episode
one tends to learn a degenerate phase pattern and stay there.

---

## Interpretability

`explain.py`:

- `integrated_gradients_for_action` — attribution for one decision, integrating
  gradients from a baseline state to the observed one along 32 steps.
- `summarize_attributions` — mean absolute attribution per named feature over
  many decisions, i.e. what the policy relies on in general rather than once.

---

## Baselines and comparison

`baselines.py` evaluates three controllers on identical seeds and episodes:

- **fixed-time** — a hold-and-switch cycle, the thing already at the junction
- **tabular Q-learning** — the discretised-state classic, included because it
  is what this problem is usually taught with
- **PPO** — greedy rollout of the trained policy

`compare.py` aggregates runs into a table with **95% confidence intervals**,
so a result is a distribution across seeds and not one lucky run.

---

## Running it

```bash
git clone https://github.com/lucenity0/Adaptive-Traffic-Signal-Controllers.git
cd Adaptive-Traffic-Signal-Controllers
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Train on the built-in simulator:

```bash
PYTHONPATH=src python -m rl_tsc.train
```

Train against SUMO — needs SUMO installed and `SUMO_HOME` set, since `traci`
ships with it:

```bash
export SUMO_HOME=/path/to/sumo
PYTHONPATH=src python -m rl_tsc.train_sumo
```

Compare against the baselines:

```bash
PYTHONPATH=src python -m rl_tsc.compare
```

Each entry point takes `--help`; seeds, episodes and curriculum mode are flags.

---

## Layout

```
src/rl_tsc/
  config.py         TrainConfig / EnvConfig — every default above lives here
  models.py         actor and critic MLPs
  ppo.py            PPOAgent: GAE, clipped update, trajectories
  env.py            MixedTrafficSignalEnv + curriculum_schedule
  sumo_adapter.py   SumoTraciAdapterEnv — TraCI loop, junction discovery
  explain.py        integrated-gradients attribution
  baselines.py      fixed-time, tabular Q-learning, PPO evaluation
  compare.py        aggregation, 95% CIs, comparison table
  train.py          training entry point (built-in simulator)
  train_sumo.py     training entry point (SUMO/TraCI)
scripts/
  generate_methodology_report.py
scenario/           a single-junction SUMO scenario to run against
requirements.txt
```

## Status

A student research project. The agent, both environments, the curriculum, the
attribution and the baseline comparison all run; the write-up around it is
kept separately while the paper is in progress.
