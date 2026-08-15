from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .config import TrainConfig
from .models import Actor, Critic


@dataclass
class Trajectory:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    last_value: torch.Tensor


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: TrainConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.actor = Actor(state_dim, action_dim, cfg.hidden_size).to(self.device)
        self.critic = Critic(state_dim, cfg.hidden_size).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def select_action(self, state: np.ndarray):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(s)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(s).squeeze(-1)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def select_action_greedy(self, state: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.actor(s)
            action = torch.argmax(logits, dim=-1)
            return int(action.item())

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        values = self.critic(states).squeeze(-1)
        return log_probs, entropy, values

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float,
        lam: float,
        last_value: torch.Tensor,
    ):
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        next_value = float(last_value.item())
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def update(self, traj: Trajectory) -> Dict[str, float]:
        states = traj.states.to(self.device)
        actions = traj.actions.to(self.device)
        old_log_probs = traj.log_probs.to(self.device)
        rewards = traj.rewards.to(self.device)
        old_values = traj.values.to(self.device)
        last_value = traj.last_value.to(self.device)

        adv, returns = self.compute_gae(rewards, old_values, self.cfg.gamma, self.cfg.gae_lambda, last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        actor_losses: List[float] = []
        critic_losses: List[float] = []
        entropies: List[float] = []

        for _ in range(self.cfg.update_epochs):
            new_log_probs, entropy, values = self.evaluate_actions(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(values, returns)
            total_actor_loss = actor_loss - self.cfg.entropy_coef * entropy

            self.actor_opt.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            (self.cfg.value_coef * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
            self.critic_opt.step()

            actor_losses.append(float(actor_loss.item()))
            critic_losses.append(float(critic_loss.item()))
            entropies.append(float(entropy.item()))

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "mean_return": float(returns.mean().item()),
        }
