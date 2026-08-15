from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import EnvConfig


@dataclass
class StepInfo:
    queue_sum: float
    max_wait: float
    emergency_active: int
    emergency_event_started: int
    emergency_served: int
    preemption_active: int
    phase: int
    phase_age: int


class MixedTrafficSignalEnv:
    """Lightweight traffic simulator with emergency preemption constraints.

    This environment mirrors the API expected by PPO training while staying runnable
    without external SUMO dependencies.
    """

    def __init__(self, cfg: EnvConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.num_lanes = 6
        self.state_dim = 10
        self.action_dim = 2

        self.queues = np.zeros(self.num_lanes, dtype=np.float32)
        self.wait = np.zeros(self.num_lanes, dtype=np.float32)

        self.phase = 0  # 0 serves EB lanes [0..2], 1 serves SB lanes [3..5]
        self.phase_age = 0

        self.av_ratio = 0.0
        self.difficulty = 0.0

        self.emergency_active = 0
        self.emergency_lane = -1
        self.emergency_wait = 0
        self.preemption_active = 0
        self.preemption_steps = 0
        self.recovery_steps_left = 0

        self.step_count = 0

    def set_curriculum(self, av_ratio: float, difficulty: float) -> None:
        self.av_ratio = float(np.clip(av_ratio, 0.0, 1.0))
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def reset(self) -> np.ndarray:
        self.queues[:] = 0.0
        self.wait[:] = 0.0
        self.phase = 0
        self.phase_age = 0
        self.emergency_active = 0
        self.emergency_lane = -1
        self.emergency_wait = 0
        self.preemption_active = 0
        self.preemption_steps = 0
        self.recovery_steps_left = 0
        self.step_count = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        self.step_count += 1
        action = int(action)

        emergency_was_active = self.emergency_active
        self._maybe_spawn_emergency()
        emergency_event_started = int(emergency_was_active == 0 and self.emergency_active == 1)

        forced_action = self._preemption_policy_override(action)
        self._apply_phase_action(forced_action)

        arrivals = self._arrivals()
        self.queues += arrivals

        served = self._service_current_phase()
        self.queues -= served
        self.queues = np.clip(self.queues, 0.0, float(self.cfg.queue_cap))

        self.wait += (self.queues > 0).astype(np.float32)
        self.wait[self.queues <= 0] = 0.0

        emergency_served = 0
        if self.emergency_active == 1 and self.emergency_lane >= 0:
            self.emergency_wait += 1
            if served[self.emergency_lane] > 0:
                emergency_served = 1
                self.emergency_active = 0
                self.emergency_lane = -1
                self.emergency_wait = 0
                self.preemption_active = 0
                self.preemption_steps = 0
                self.recovery_steps_left = self.cfg.preemption_recovery_steps

        reward = self._compute_reward(served, emergency_served, forced_action != action)

        done = False
        info_obj = StepInfo(
            queue_sum=float(self.queues.sum()),
            max_wait=float(self.wait.max()),
            emergency_active=int(self.emergency_active),
            emergency_event_started=int(emergency_event_started),
            emergency_served=int(emergency_served),
            preemption_active=int(self.preemption_active),
            phase=int(self.phase),
            phase_age=int(self.phase_age),
        )

        return self._get_state(), reward, done, info_obj.__dict__

    def _get_state(self) -> np.ndarray:
        q_norm = self.queues / float(self.cfg.queue_cap)
        phase_norm = np.array([float(self.phase)], dtype=np.float32)
        phase_age_norm = np.array(
            [min(self.phase_age, self.cfg.max_green_steps) / float(self.cfg.max_green_steps)], dtype=np.float32
        )
        av_ratio = np.array([self.av_ratio], dtype=np.float32)
        emergency = np.array([float(self.emergency_active)], dtype=np.float32)
        state = np.concatenate([q_norm, phase_norm, phase_age_norm, av_ratio, emergency], axis=0)
        return state.astype(np.float32)

    def _arrivals(self) -> np.ndarray:
        demand_scale = 1.0 + 0.8 * self.difficulty
        av_smoothing = 1.0 - 0.2 * self.av_ratio
        lam = self.cfg.base_arrival_rate * demand_scale * av_smoothing

        arrivals = self.np_rng.poisson(lam=lam, size=self.num_lanes).astype(np.float32)
        return np.clip(arrivals, 0.0, 4.0)

    def _service_current_phase(self) -> np.ndarray:
        served = np.zeros(self.num_lanes, dtype=np.float32)
        active_lanes = [0, 1, 2] if self.phase == 0 else [3, 4, 5]

        base = self.cfg.base_service_rate
        av_efficiency = 1.0 + 0.25 * self.av_ratio
        capacity = base * av_efficiency

        for lane in active_lanes:
            served[lane] = min(self.queues[lane], capacity)

        return served

    def _apply_phase_action(self, action: int) -> None:
        can_switch = self.phase_age >= self.cfg.min_green_steps
        must_switch = self.phase_age >= self.cfg.max_green_steps

        if must_switch:
            self.phase = 1 - self.phase
            self.phase_age = 0
            return

        if action == 1 and can_switch:
            self.phase = 1 - self.phase
            self.phase_age = 0
        else:
            self.phase_age += 1

        if self.recovery_steps_left > 0:
            self.recovery_steps_left -= 1

    def _maybe_spawn_emergency(self) -> None:
        if self.emergency_active == 1:
            return

        if self.rng.random() < self.cfg.emergency_prob:
            self.emergency_active = 1
            self.emergency_lane = self.rng.randint(0, self.num_lanes - 1)
            self.emergency_wait = 0

    def _preemption_policy_override(self, action: int) -> int:
        if self.emergency_active == 0:
            return action

        target_phase = 0 if self.emergency_lane in (0, 1, 2) else 1

        if self.phase == target_phase:
            self.preemption_active = 1
            self.preemption_steps += 1
            if self.preemption_steps >= self.cfg.preemption_cap_steps:
                self.preemption_active = 0
                self.preemption_steps = 0
            return 0

        self.preemption_active = 1
        self.preemption_steps += 1
        if self.preemption_steps <= self.cfg.preemption_cap_steps:
            return 1

        self.preemption_active = 0
        self.preemption_steps = 0
        return action

    def _compute_reward(self, served: np.ndarray, emergency_served: int, forced_switch: bool) -> float:
        queue_norm = float(self.queues.sum()) / max(1.0, float(self.cfg.queue_cap * self.num_lanes))
        wait_norm = float(self.wait.max()) / max(1.0, float(self.cfg.max_green_steps))

        queue_penalty = -self.cfg.queue_penalty_coef * queue_norm
        max_wait_penalty = -self.cfg.wait_penalty_coef * wait_norm

        switch_cost = -self.cfg.switch_penalty if forced_switch else 0.0

        emergency_term = 0.0
        if self.emergency_active == 1:
            emergency_term -= self.cfg.emergency_delay_penalty
        if emergency_served == 1:
            emergency_term += self.cfg.emergency_served_bonus

        reward = queue_penalty + max_wait_penalty + switch_cost + emergency_term
        return float(np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip))


def curriculum_schedule(episode: int, total_episodes: int, mode: str = "linear") -> Tuple[float, float]:
    if mode == "off":
        return 0.0, 0.0
    if mode == "max":
        return 1.0, 1.0

    x = episode / max(1, total_episodes - 1)
    av_ratio = x
    difficulty = 0.2 + 0.8 * math.sqrt(x)
    return float(np.clip(av_ratio, 0.0, 1.0)), float(np.clip(difficulty, 0.0, 1.0))
