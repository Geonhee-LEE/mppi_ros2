"""
Disturbance Profiles for Dynamic World Simulation.

시간에 따라 변하는 외란을 주입하여 고정 모델 vs 적응형 모델(MAML)의 차이를 부각.

설계 원칙:
  - 힘은 속도 상태(v, ω)에 적용 → 컨트롤러 내부 모델의 예측 오류 발생
    (위치 교란은 피드백 루프가 즉시 보상하므로 모델 품질 무관)
  - 마찰 계수는 시간에 따라 다중 전환 → 고정 모델은 구조적 대응 불가
  - intensity=0.0 → 기존 동작과 동일, intensity=1.0 → 매우 도전적

Profiles:
    - WindGustDisturbance:      간헐적 풍하중 → 속도 교란 (unmodeled acceleration)
    - TerrainChangeDisturbance: c_v, c_omega 다중 단계 변동 (마찰 계수 반복 변화)
    - SinusoidalDisturbance:    주기적 가속도 외란 → 속도 교란
    - CombinedDisturbance:      위 3개 합성

Usage:
    profile = CombinedDisturbance(intensity=0.5, seed=42)
    force = profile.get_force(t=3.0, state=np.zeros(5))
    param_delta = profile.get_param_delta(t=3.0)
"""

import numpy as np
from abc import ABC, abstractmethod


class DisturbanceProfile(ABC):
    """외란 프로필 추상 기저 클래스."""

    def __init__(self, intensity: float = 0.3, seed: int = 42):
        """
        Args:
            intensity: 외란 강도 (0.0~1.0). --noise CLI 값에 매핑.
            seed: 재현성을 위한 랜덤 시드.
        """
        self.intensity = np.clip(intensity, 0.0, 1.0)
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    @abstractmethod
    def get_force(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        시간 t, 상태 state에서의 5D 외란 벡터.

        force[3] (dv 가속도), force[4] (dω 각가속도)에 적용하여
        컨트롤러 내부 모델이 예측하지 못하는 unmodeled acceleration 생성.

        Returns:
            np.ndarray shape (5,): [dx, dy, dθ, dv, dω] 외란.
        """
        pass

    def get_param_delta(self, t: float) -> dict:
        """
        시간 t에서의 동역학 파라미터 변경분.

        Returns:
            dict with optional keys 'delta_c_v', 'delta_c_omega'.
            빈 dict이면 파라미터 변동 없음.
        """
        return {}


class WindGustDisturbance(DisturbanceProfile):
    """
    간헐적 풍하중 — 속도(v, ω)에 돌풍 가속도 적용.

    위치가 아닌 속도를 교란하면:
    - 컨트롤러 내부 모델은 dv/dt = k*(v_cmd-v) - c_v*v 로 예측
    - 실제는 dv/dt = k*(v_cmd-v) - c_v*v + wind_accel
    - 30-step 호라이즌 전체에서 예측 오류 누적 → 모델 품질이 중요해짐
    """

    def __init__(self, intensity: float = 0.3, seed: int = 42,
                 num_gusts: int = 8, max_accel: float = 3.0,
                 duration_range: tuple = (15.0,)):
        super().__init__(intensity, seed)
        self.max_accel = max_accel  # m/s² 단위 가속도

        total_duration = duration_range[0] if duration_range else 15.0
        self.gusts = []
        for _ in range(num_gusts):
            onset = self._rng.uniform(0.5, total_duration - 1.0)
            gust_dur = self._rng.uniform(0.5, 2.5)
            angle = self._rng.uniform(0, 2 * np.pi)
            strength = self._rng.uniform(0.5, 1.0)
            # 선속도 vs 각속도 비율
            omega_frac = self._rng.uniform(-0.5, 0.5)
            self.gusts.append({
                "onset": onset,
                "duration": gust_dur,
                "angle": angle,
                "strength": strength,
                "omega_frac": omega_frac,
            })

    def get_force(self, t: float, state: np.ndarray) -> np.ndarray:
        force = np.zeros(5)
        for g in self.gusts:
            if g["onset"] <= t <= g["onset"] + g["duration"]:
                phase = (t - g["onset"]) / g["duration"]
                envelope = np.sin(np.pi * phase)
                accel = self.intensity * self.max_accel * g["strength"] * envelope
                # 로봇 바디 프레임 기준 가속도 → 속도 상태에 적용
                theta = state[2] if len(state) >= 3 else 0.0
                # 풍향을 바디 프레임으로 변환하여 v (전진)에 투영
                wind_angle_body = g["angle"] - theta
                force[3] += accel * np.cos(wind_angle_body)  # dv (선가속도)
                force[4] += accel * g["omega_frac"]           # dω (각가속도)
        return force


class TerrainChangeDisturbance(DisturbanceProfile):
    """
    지형 변화 — c_v, c_omega가 시간에 따라 다중 단계 변동.

    단일 전환이 아닌 여러 번 변동하여:
    - 고정 파라미터 모델은 한 번도 맞지 않음
    - MAML은 매 전환마다 재적응하여 추적 가능
    """

    def __init__(self, intensity: float = 0.3, seed: int = 42,
                 sim_duration: float = 20.0,
                 num_transitions: int = 3,
                 max_delta_c_v: float = 1.5,
                 max_delta_c_omega: float = 0.8):
        super().__init__(intensity, seed)
        self.sim_duration = sim_duration
        self.max_delta_c_v = max_delta_c_v
        self.max_delta_c_omega = max_delta_c_omega

        # 다중 전환점 생성 (시뮬레이션 시간을 구간으로 분할)
        self.transitions = []
        segment = sim_duration / (num_transitions + 1)
        for i in range(num_transitions):
            t_switch = segment * (i + 1)
            # 각 전환에서 마찰이 크게 바뀜
            dc_v = self._rng.uniform(0.5, 1.0) * max_delta_c_v
            dc_omega = self._rng.uniform(0.3, 1.0) * max_delta_c_omega
            # 교대로 증가/감소 (고정 모델은 어떤 값에도 맞출 수 없음)
            sign = 1.0 if i % 2 == 0 else -0.3
            self.transitions.append({
                "time": t_switch,
                "delta_c_v": dc_v * sign,
                "delta_c_omega": dc_omega * sign,
            })

    def get_force(self, t: float, state: np.ndarray) -> np.ndarray:
        return np.zeros(5)

    def get_param_delta(self, t: float) -> dict:
        # 활성화된 전환들의 누적 효과 (sigmoid 전환)
        total_dc_v = 0.0
        total_dc_omega = 0.0
        steepness = 3.0
        for tr in self.transitions:
            if t >= tr["time"] - 1.0:  # sigmoid가 시작되는 시점
                s = 1.0 / (1.0 + np.exp(-steepness * (t - tr["time"])))
                total_dc_v += self.intensity * tr["delta_c_v"] * s
                total_dc_omega += self.intensity * tr["delta_c_omega"] * s

        if abs(total_dc_v) > 1e-8 or abs(total_dc_omega) > 1e-8:
            return {
                "delta_c_v": total_dc_v,
                "delta_c_omega": total_dc_omega,
            }
        return {}


class SinusoidalDisturbance(DisturbanceProfile):
    """
    주기적 가속도 외란 — 속도(v, ω)에 sin/cos 기반 교란.

    컨트롤러가 예측하지 못하는 주기적 가속도를 속도에 적용하여
    모델-세계 간 예측 오차를 체계적으로 발생시킴.
    """

    def __init__(self, intensity: float = 0.3, seed: int = 42,
                 v_amplitude: float = 2.0, omega_amplitude: float = 1.0,
                 frequency: float = 0.3):
        super().__init__(intensity, seed)
        self.v_amplitude = v_amplitude      # 선가속도 진폭 (m/s²)
        self.omega_amplitude = omega_amplitude  # 각가속도 진폭 (rad/s²)
        self.frequency = frequency  # Hz
        self._phase_v = self._rng.uniform(0, 2 * np.pi)
        self._phase_omega = self._rng.uniform(0, 2 * np.pi)

    def get_force(self, t: float, state: np.ndarray) -> np.ndarray:
        force = np.zeros(5)
        omega = 2.0 * np.pi * self.frequency
        force[3] = self.intensity * self.v_amplitude * np.sin(omega * t + self._phase_v)
        force[4] = self.intensity * self.omega_amplitude * np.cos(omega * t + self._phase_omega)
        return force


class CombinedDisturbance(DisturbanceProfile):
    """
    WindGust + TerrainChange + Sinusoidal 합성.

    세 가지 외란을 동시에 적용하여 가장 도전적인 환경 생성:
    - Wind: 간헐적 속도 교란 (unmodeled acceleration)
    - Terrain: 마찰 계수 다중 전환 (dynamics parameter shift)
    - Sine: 주기적 속도 교란 (persistent model mismatch)
    """

    def __init__(self, intensity: float = 0.3, seed: int = 42,
                 sim_duration: float = 20.0):
        super().__init__(intensity, seed)
        self.wind = WindGustDisturbance(
            intensity=intensity, seed=seed,
            num_gusts=8, max_accel=3.0,
            duration_range=(sim_duration,),
        )
        self.terrain = TerrainChangeDisturbance(
            intensity=intensity, seed=seed + 1,
            sim_duration=sim_duration,
            num_transitions=3,
            max_delta_c_v=1.5,
            max_delta_c_omega=0.8,
        )
        self.sinusoidal = SinusoidalDisturbance(
            intensity=intensity, seed=seed + 2,
            v_amplitude=1.5, omega_amplitude=0.8,
            frequency=0.25,
        )

    def get_force(self, t: float, state: np.ndarray) -> np.ndarray:
        return (self.wind.get_force(t, state)
                + self.sinusoidal.get_force(t, state))

    def get_param_delta(self, t: float) -> dict:
        return self.terrain.get_param_delta(t)


def create_disturbance(disturbance_type: str, intensity: float,
                       seed: int = 42, sim_duration: float = 20.0):
    """
    외란 프로필 팩토리 함수.

    Args:
        disturbance_type: "none", "wind", "terrain", "sine", "combined"
        intensity: 외란 강도 (0.0~1.0)
        seed: 랜덤 시드
        sim_duration: 시뮬레이션 총 시간 (WindGust 이벤트 배치용)

    Returns:
        DisturbanceProfile or None (if "none")
    """
    if disturbance_type == "none" or intensity <= 0.0:
        return None
    elif disturbance_type == "wind":
        return WindGustDisturbance(intensity=intensity, seed=seed,
                                   duration_range=(sim_duration,))
    elif disturbance_type == "terrain":
        return TerrainChangeDisturbance(intensity=intensity, seed=seed,
                                         sim_duration=sim_duration)
    elif disturbance_type == "sine":
        return SinusoidalDisturbance(intensity=intensity, seed=seed)
    elif disturbance_type == "combined":
        return CombinedDisturbance(intensity=intensity, seed=seed,
                                    sim_duration=sim_duration)
    else:
        raise ValueError(f"Unknown disturbance type: {disturbance_type}")
