from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SumoAdapterConfig:
    sumo_cfg_path: str = "scenario.sumocfg"
    tls_id: Optional[str] = None
    sumo_binary: str = "sumo"
    gui_settings_path: Optional[str] = None
    use_gui: bool = False
    max_controlled_lanes: int = 6
    min_green_steps: int = 8
    max_green_steps: int = 45
    phase_sequence: Optional[Tuple[int, int]] = None
    queue_norm_cap: float = 60.0
    emergency_vclass: str = "emergency"


class SumoTraciAdapterEnv:
    """SUMO-TraCI adapter environment for arbitrary single-junction scenarios.

    This class intentionally keeps a minimal API compatible with the PPO training loop:
    - reset() -> state
    - step(action) -> next_state, reward, done, info

    Action mapping:
    - 0: hold current phase
    - 1: switch to next phase in phase_sequence
    """

    def __init__(self, cfg: SumoAdapterConfig):
        self.cfg = cfg
        self.phase_idx = 0
        self.phase_age = 0
        self.current_step = 0

        self.state_dim = 10
        self.action_dim = 2

        self._traci = None
        self._connected = False
        self._controlled_lanes: List[str] = []
        self._phase_sequence: Tuple[int, int] = (0, 1)

    @property
    def available(self) -> bool:
        try:
            import traci  # noqa: F401

            return True
        except Exception:
            return False

    def connect(self, sumo_binary: Optional[str] = None) -> None:
        if self._connected:
            return

        try:
            import traci
        except Exception as exc:
            raise RuntimeError(
                "TraCI is not available. Install SUMO and ensure Python tools are on PYTHONPATH."
            ) from exc

        binary = sumo_binary or self.cfg.sumo_binary
        traci.start(self._build_sumo_args(binary))
        self._traci = traci
        self._connected = True
        self._discover_control_target()

    def _reload(self) -> None:
        if not self._connected or self._traci is None:
            return
        try:
            self._traci.load(self._build_reload_args())
            self._discover_control_target()
        except Exception:
            self.close()
            self.connect()

    def close(self) -> None:
        if self._connected and self._traci is not None:
            self._traci.close()
            self._connected = False
            self._traci = None

    def reset(self) -> np.ndarray:
        if not self._connected:
            self.connect()
        else:
            self._reload()

        self.phase_idx = 0
        self.phase_age = 0
        self.current_step = 0
        self._set_phase(self._phase_sequence[self.phase_idx])
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if not self._connected:
            raise RuntimeError("SUMO adapter is not connected. Call reset() first.")

        self.current_step += 1
        self._apply_action(int(action))

        self._traci.simulationStep()

        queues = self._read_detector_queues()
        queue_sum = float(np.sum(queues))
        max_wait = self._estimate_max_wait(queues)
        emergency_active = float(self._detect_emergency_active())

        reward = -queue_sum - 0.1 * max_wait - 0.25 * emergency_active

        info = {
            "queue_sum": queue_sum,
            "max_wait": max_wait,
            "emergency_active": emergency_active,
            "emergency_served": 0.0,
            "preemption_active": 0.0,
            "phase": float(self.phase_idx),
            "phase_age": float(self.phase_age),
        }

        done = False
        return self._get_state(), float(reward), done, info

    def _apply_action(self, action: int) -> None:
        can_switch = self.phase_age >= self.cfg.min_green_steps
        must_switch = self.phase_age >= self.cfg.max_green_steps

        if must_switch or (action == 1 and can_switch):
            self.phase_idx = 1 - self.phase_idx
            self._set_phase(self._phase_sequence[self.phase_idx])
            self.phase_age = 0
        else:
            self.phase_age += 1

    def _set_phase(self, phase_id: int) -> None:
        self._traci.trafficlight.setPhase(self.cfg.tls_id, int(phase_id))

    def _read_lane_queues(self) -> np.ndarray:
        values: List[float] = []
        for lane_id in self._controlled_lanes:
            try:
                v = float(self._traci.lane.getLastStepHaltingNumber(lane_id))
            except Exception:
                v = 0.0
            values.append(v)
        while len(values) < self.cfg.max_controlled_lanes:
            values.append(0.0)
        return np.array(values[: self.cfg.max_controlled_lanes], dtype=np.float32)

    def _detect_emergency_active(self) -> bool:
        for vid in self._traci.vehicle.getIDList():
            try:
                if self._traci.vehicle.getVehicleClass(vid) == self.cfg.emergency_vclass:
                    return True
            except Exception:
                continue
        return False

    def _estimate_max_wait(self, queues: np.ndarray) -> float:
        return float(np.max(queues))

    def _discover_control_target(self) -> None:
        tls_ids = list(self._traci.trafficlight.getIDList())
        if not tls_ids:
            raise RuntimeError("No traffic lights found in SUMO scenario.")

        if self.cfg.tls_id and self.cfg.tls_id in tls_ids:
            self.cfg.tls_id = self.cfg.tls_id
        else:
            self.cfg.tls_id = tls_ids[0]

        lanes = list(dict.fromkeys(self._traci.trafficlight.getControlledLanes(self.cfg.tls_id)))
        self._controlled_lanes = lanes[: self.cfg.max_controlled_lanes]

        if self.cfg.phase_sequence is not None:
            self._phase_sequence = self.cfg.phase_sequence
            return

        candidates: List[int] = []
        try:
            programs = self._traci.trafficlight.getAllProgramLogics(self.cfg.tls_id)
            if programs:
                phases = programs[0].phases
                for idx, p in enumerate(phases):
                    state = getattr(p, "state", "")
                    if any(ch in state for ch in ("G", "g")):
                        candidates.append(idx)
        except Exception:
            pass

        unique = []
        for idx in candidates:
            if idx not in unique:
                unique.append(idx)
        if len(unique) >= 2:
            self._phase_sequence = (unique[0], unique[1])
        elif len(unique) == 1:
            self._phase_sequence = (unique[0], unique[0])
        else:
            self._phase_sequence = (0, 1)

    def _get_state(self) -> np.ndarray:
        queues = self._read_lane_queues()
        q_norm = np.clip(queues / self.cfg.queue_norm_cap, 0.0, 1.0)

        phase_val = np.array([float(self.phase_idx)], dtype=np.float32)
        phase_age_val = np.array(
            [min(self.phase_age, self.cfg.max_green_steps) / float(self.cfg.max_green_steps)], dtype=np.float32
        )

        # Placeholders retained for compatibility with lightweight env shape.
        av_ratio = np.array([0.0], dtype=np.float32)
        emergency = np.array([float(self._detect_emergency_active())], dtype=np.float32)

        state = np.concatenate([q_norm, phase_val, phase_age_val, av_ratio, emergency], axis=0)
        return state.astype(np.float32)

    def _build_sumo_args(self, binary: str) -> List[str]:
        self._normalize_gui_display(binary)
        args = [binary, "-c", self.cfg.sumo_cfg_path]
        gui_settings_path = self._resolve_gui_settings_path(binary)
        if gui_settings_path:
            args.extend(["--gui-settings-file", gui_settings_path])
        return args

    def _build_reload_args(self) -> List[str]:
        args = ["-c", self.cfg.sumo_cfg_path]
        gui_settings_path = self._resolve_gui_settings_path(self.cfg.sumo_binary)
        if gui_settings_path:
            args.extend(["--gui-settings-file", gui_settings_path])
        return args

    def _resolve_gui_settings_path(self, binary: str) -> Optional[str]:
        if "gui" not in os.path.basename(binary):
            return None

        if self.cfg.gui_settings_path:
            return self.cfg.gui_settings_path

        cfg_path = os.path.abspath(self.cfg.sumo_cfg_path)
        cfg_dir = os.path.dirname(cfg_path)

        configured_view = self._read_gui_settings_from_sumocfg(cfg_path)
        if configured_view:
            candidate = configured_view
            if not os.path.isabs(candidate):
                candidate = os.path.join(cfg_dir, candidate)
            if os.path.exists(candidate):
                return candidate

        stem = os.path.splitext(os.path.basename(cfg_path))[0]
        candidate = os.path.join(cfg_dir, f"{stem}.view.xml")
        if os.path.exists(candidate):
            return candidate

        return None

    def _read_gui_settings_from_sumocfg(self, cfg_path: str) -> Optional[str]:
        try:
            root = ET.parse(cfg_path).getroot()
        except Exception:
            return None

        for section in root.findall("gui_only"):
            for node in section.findall("gui-settings-file"):
                value = node.attrib.get("value")
                if value:
                    return value
        return None

    def _normalize_gui_display(self, binary: str) -> None:
        if "gui" not in os.path.basename(binary):
            return
        if os.uname().sysname != "Darwin":
            return

        display = os.environ.get("DISPLAY", "")
        if display.startswith("/private/tmp/") or display.startswith("/tmp/"):
            # SUMO documents blank macOS windows with XQuartz until DISPLAY is reset.
            os.environ["DISPLAY"] = ":0.0"
        os.environ.setdefault("LIBGL_ALWAYS_INDIRECT", "1")
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
