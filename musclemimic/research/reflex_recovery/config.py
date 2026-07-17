"""Typed configuration for reflex-recovery experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PerturbationConfig:
    """Erroneous exoskeleton joint-torque pulse configuration (SI units)."""

    enabled: bool = False
    joint: str = "ankle"
    side: str = "right"
    direction: int = 1
    magnitude_nm: float = 0.0
    magnitude_fraction_peak: float | None = None
    duration_s: float = 0.05
    onset_mode: str = "time"
    onset_time_s: float | None = 1.0
    onset_phase: float | None = None
    phase_tolerance: float = 0.02
    waveform: str = "rectangular"
    randomize_joint: bool = False
    randomize_side: bool = False
    randomize_direction: bool = False
    randomize_magnitude: bool = False
    randomize_duration: bool = False
    randomize_phase: bool = False

    def __post_init__(self) -> None:
        if self.joint not in {"hip", "knee", "ankle"}:
            raise ValueError(f"Unsupported joint: {self.joint}")
        if self.side not in {"left", "right"}:
            raise ValueError(f"Unsupported side: {self.side}")
        if self.direction not in {-1, 1}:
            raise ValueError("direction must be -1 or 1")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.waveform not in {"rectangular", "half_sine", "triangular"}:
            raise ValueError(f"Unsupported waveform: {self.waveform}")
        if self.onset_mode not in {"time", "phase", "heel_strike", "stance_percentage"}:
            raise ValueError(f"Unsupported onset mode: {self.onset_mode}")
        if self.onset_mode == "time" and self.onset_time_s is None:
            raise ValueError("onset_time_s is required for time-triggered perturbations")
        if self.onset_mode in {"phase", "stance_percentage"} and self.onset_phase is None:
            raise ValueError(f"onset_phase is required for {self.onset_mode} triggering")
        if self.onset_phase is not None and not 0.0 <= self.onset_phase <= 1.0:
            raise ValueError("onset_phase must lie in [0, 1]")


@dataclass(frozen=True)
class ReflexConfig:
    """Limits and timing for the grouped reflex residual."""

    enabled: bool = False
    scale: float = 1.0
    delay_s: float = 0.0
    group_limit: float = 0.25
    muscle_limit: float = 0.25
    rate_limit_per_s: float = 5.0


@dataclass(frozen=True)
class RecoveryGateConfig:
    """Continuous instability-to-recovery activation configuration."""

    risk_threshold: float = 1.0
    sigmoid_gain: float = 6.0
    low_pass_fraction: float = 0.2
    off_hysteresis: float = 0.1


@dataclass(frozen=True)
class ActionComposerConfig:
    """Normalized policy-action composition limits."""

    reflex_scale: float = 1.0
    recovery_scale: float = 1.0
    residual_limit: float = 0.5
    rate_limit_per_s: float = 10.0


@dataclass(frozen=True)
class ReflexRecoveryConfig:
    """Top-level configuration used by scripts and environment adapters."""

    control_dt_s: float = 0.01
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    reflex: ReflexConfig = field(default_factory=ReflexConfig)
    gate: RecoveryGateConfig = field(default_factory=RecoveryGateConfig)
    composer: ActionComposerConfig = field(default_factory=ActionComposerConfig)
