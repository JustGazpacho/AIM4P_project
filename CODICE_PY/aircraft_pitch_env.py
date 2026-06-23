"""
Aircraft Pitch Stabilization - Extended Environment
====================================================
Tracks a dynamic 3-D flight path instead of a fixed pitch target.

Observation (10 elementi):
    0  theta_norm        pitch angle / THETA_LIMIT              in [-1, 1]
    1  theta_dot_norm    pitch rate  / 5.0                       in [-1, 1]
    2  alt_error_norm    (actual - target alt) / 5000            in [-1, 1]
    3  roc_error_norm    (actual ROC - target ROC) / 20.0        in [-1, 1]
    4  heading_error     wrapped heading error / 180 deg         in [-1, 1]
    5  airspeed_norm     (airspeed - 225) / 75                   in [-1, 1]
    6  density_norm      (density - 0.5) / 0.5                   in [-1, 1]
    7  time_to_wp_norm   time_to_next_waypoint / MAX_SEGMENT_S    in [0, 1]
    8  target_roc_norm   target rate of climb / 20.0             in [-1, 1]
    9  disturbance_norm  total atmospheric torque / 3.0          in [-2, 2]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import argparse

from flight_path import FlightPath
from atmosphere import AtmosphereModel


class AircraftPitchEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}


    I  = 1.0
    D  = 0.4
    K  = 0.2
    dt = 0.05

    ALT_PITCH_COUPLING = 0.0002
    HDG_ROLL_COUPLING  = 0.0003

    ALT_PITCH_BIAS_MAX = 0.3
    HDG_ROLL_LOAD_MAX  = 0.1


    N_ACTIONS = 11
    T_MAX     = 5.46  # calibrated: authority margin @ 99.7 pct, severity=severe, n=1000 ep



    MAX_STEPS     = 2400
    THETA_LIMIT   = np.radians(60)
    STABLE_THRESH = np.radians(2)

    DIVERGENCE_PROXIMITY_FRAC = 0.3
    DIVERGENCE_PENALTY_GAIN   = 5.0

    SOFT_STABLE_BONUS_GAIN  = 0.5
    SOFT_STABLE_BONUS_DECAY = 4.0

    DISTURBANCE_OBS_SCALE = 3.0
    DISTURBANCE_OBS_CLIP  = 2.0

    ALT_TOLERANCE_M   = 50.0
    HDG_TOLERANCE_DEG = 10.0
    MAX_ROC_MS        = 20.0

    DQN_LEARNING_STARTS = 24_000

    ANGLE_PENALTY_THRESHOLD_DEG = 20.0
    ANGLE_PENALTY_CUBIC_DEG     = 30.0
    ANGLE_PENALTY_GAIN_QUAD     = 2.0
    ANGLE_PENALTY_GAIN_CUBIC    = 4.0

    _SEVERITIES = ["light", "moderate", "severe"]
    SEVERITY_REWARD_GAIN = {
        "light":    1.0,
        "moderate": 1.4,
        "severe":   2.0,
    }
    _DEFAULT_SEVERITY_WEIGHTS = [0.20, 0.45, 0.35]

    def __init__(self, render_mode=None,
                 turbulence_severity: str = "moderate",
                 n_path_waypoints: int = 6,
                 initial_max_steps: int = None):
        """
        Initialise the environment.

        Args:
            render_mode:          "human" for live plot, "rgb_array" for frame capture, None for headless.
            turbulence_severity:  "light", "moderate", "severe", or "random" (sampled each episode).
            n_path_waypoints:     Number of waypoints in the generated flight path.
            initial_max_steps:    Episode length in steps; defaults to MAX_STEPS (2400).
                                Can be changed at runtime via set_max_steps().
        """
        super().__init__()
        self.render_mode         = render_mode
        self.turbulence_severity = turbulence_severity
        self.n_path_waypoints    = n_path_waypoints

        self._effective_max_steps = (
            int(initial_max_steps) if initial_max_steps is not None
            else self.MAX_STEPS
        )
        self._episode_duration = self._effective_max_steps * self.dt
        self._pending_max_steps = None

        _cruise = max(FlightPath.MIN_CRUISE_S,
                      FlightPath.CRUISE_FRACTION * self._episode_duration)
        self._ttw_norm = (self._episode_duration - _cruise) / max(n_path_waypoints, 1)

        low = np.array([
            -1, -1, -1, -1, -1,
            -1, -1,  0, -1, -self.DISTURBANCE_OBS_CLIP
        ], dtype=np.float32)

        high = np.array([
            1,  1,  1,  1,  1,
            1,  1,  1,  1,  self.DISTURBANCE_OBS_CLIP
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = spaces.Discrete(self.N_ACTIONS)
        self._torques          = np.linspace(-self.T_MAX, self.T_MAX, self.N_ACTIONS)

        self._severity_weights = list(self._DEFAULT_SEVERITY_WEIGHTS)

        init_severity = (
            "moderate" if turbulence_severity == "random" else turbulence_severity
        )

        self._flight_path = FlightPath(
            n_waypoints=n_path_waypoints,
            duration=self._episode_duration,
        )
        self._atmosphere = AtmosphereModel(
            dt=self.dt,
            turbulence_severity=init_severity,
        )
        self._current_severity = init_severity

        self._state       = None
        self._steps       = 0
        self._time        = 0.0
        self._history     = []
        self._prev_torque = 0.0
        self._prev_altitude = 0.0

        self.altitude = 0.0
        self.heading  = 0.0
        self.airspeed = 200.0
        self._last_disturbance = 0.0

        self._renderer = None

    def set_max_steps(self, n: int) -> None:
        """
        Schedule an episode-length change to take effect at the next reset().
        The change is deferred so that an in-progress episode is not interrupted.
        Intended for progressive episode-length curriculum callbacks.
        """
        n = int(np.clip(n, 1, self.MAX_STEPS))
        if n == self._effective_max_steps and self._pending_max_steps is None:
            return
        self._pending_max_steps = n

    def _apply_pending_max_steps(self) -> None:
        """
        Apply a pending set_max_steps() request at the start of a new episode.
        Rebuilds the flight path spline to match the new duration.
        No-op if no change is pending or the value is unchanged.
        """
        if self._pending_max_steps is None:
            return
        n = self._pending_max_steps
        self._pending_max_steps = None

        if n == self._effective_max_steps:
            return

        self._effective_max_steps = n
        self._episode_duration    = n * self.dt

        _cruise = max(FlightPath.MIN_CRUISE_S,
                      FlightPath.CRUISE_FRACTION * self._episode_duration)
        self._ttw_norm = (self._episode_duration - _cruise) / max(self.n_path_waypoints, 1)

        self._flight_path = FlightPath(
            n_waypoints=self.n_path_waypoints,
            duration=self._episode_duration,
        )

    def set_turbulence(self, severity: str) -> None:
        """
        Immediately switch turbulence severity for the current and future episodes.
        Unlike set_severity_weights(), this bypasses random sampling and forces
        a specific level (e.g. for deterministic evaluation).
        """
        self._current_severity = severity
        self._atmosphere.turbulence._apply_severity(severity)

    def set_severity_weights(self, weights) -> None:
        """
        Update the probability distribution over turbulence severities used when
        turbulence_severity="random". Weights are normalised internally.
        Degenerate inputs (all zeros) are silently ignored.

        Args:
            weights: array-like of length 3 - [light, moderate, severe].
        """
        weights = np.asarray(weights, dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            return  
        self._severity_weights = (weights / total).tolist()

    @classmethod
    def calibrate_t_max(cls, severity: str = "severe", episodes: int = 100,
                         percentile: float = 99.7, probe_tmax: float = 50.0,
                         verbose: bool = True) -> dict:
        """
        Calibrate T_MAX using the authority-margin principle
        (MIL-HDBK-1797 / NASA gust-load-alleviation): control authority must
        cover the atmospheric disturbance at the requested percentile (default
        99.7 %, i.e. 3-sigma), not just the mean or RMS.

        Method:
        1. Run `episodes` episodes at `severity` with T_MAX temporarily raised
            to `probe_tmax` so the controller never saturates and does not
            prematurely truncate data collection. Atmospheric disturbances are
            generated independently of the chosen action, so their distribution
            is unaffected by T_MAX.
        2. Collect total_torque = gust + shear + thermal at every step.
        3. Compute the requested percentile of |total_torque|.

        Returns a dict of statistics and the recommended T_MAX.
        Does NOT modify the class attribute T_MAX; the caller decides whether
        to apply the result.

        Usage:
            python aircraft_pitch_env.py --calibrate --severity severe --episodes 1000
        """
        env = cls(turbulence_severity=severity)
        original_t_max = env.T_MAX
        env.T_MAX = probe_tmax
        env._torques = np.linspace(-env.T_MAX, env.T_MAX, env.N_ACTIONS)

        total, gust, shear, thermal = [], [], [], []
        for ep in range(episodes):
            env.reset(seed=ep)
            done = False
            while not done:
                action = env.action_space.sample()
                _, _, term, trunc, info = env.step(action)
                total.append(info["gust_torque"] + info["shear_torque"] + info["thermal_torque"])
                gust.append(info["gust_torque"])
                shear.append(info["shear_torque"])
                thermal.append(info["thermal_torque"])
                done = term or trunc
        env.close()

        total = np.asarray(total); gust = np.asarray(gust)
        shear = np.asarray(shear); thermal = np.asarray(thermal)
        abs_total = np.abs(total)

        result = {
            "severity":          severity,
            "n_steps":           len(total),
            "rms_total":         float(np.sqrt(np.mean(total**2))),
            "median_abs_total":  float(np.percentile(abs_total, 50.0)),
            "p95_abs_total":     float(np.percentile(abs_total, 95.0)),
            "p99_abs_total":     float(np.percentile(abs_total, 99.0)),
            "p997_abs_total":    float(np.percentile(abs_total, percentile)),
            "max_abs_total":     float(abs_total.max()),
            "rms_gust":          float(np.sqrt(np.mean(gust**2))),
            "rms_shear":         float(np.sqrt(np.mean(shear**2))),
            "rms_thermal":       float(np.sqrt(np.mean(thermal**2))),
            "recommended_t_max": float(np.percentile(abs_total, percentile)),
            "percentile_used":   percentile,
            "original_t_max":    float(original_t_max),
        }

        if verbose:
            print(f"\n{'='*55}")
            print(f"  T_MAX CALIBRATION  (severity={severity}, n={result['n_steps']} steps)")
            print(f"{'='*55}")
            print(f"  total_torque RMS         : {result['rms_total']:.3f} N*m")
            print(f"  total_torque median       : {result['median_abs_total']:.3f} N*m")
            print(f"  total_torque 95th pct     : {result['p95_abs_total']:.3f} N*m")
            print(f"  total_torque 99th pct     : {result['p99_abs_total']:.3f} N*m")
            print(f"  total_torque {percentile}th pct   : {result['p997_abs_total']:.3f} N*m  <-- target")
            print(f"  total_torque max observed : {result['max_abs_total']:.3f} N*m")
            print(f"  -- breakdown (RMS) --")
            print(f"  gust_torque    RMS        : {result['rms_gust']:.3f} N*m")
            print(f"  shear_torque   RMS        : {result['rms_shear']:.3f} N*m")
            print(f"  thermal_torque RMS        : {result['rms_thermal']:.3f} N*m")
            print(f"{'='*55}")
            print(f"  Current  T_MAX = {result['original_t_max']:.2f}")
            print(f"  Suggested T_MAX = {result['recommended_t_max']:.2f}"
                  f"  (authority @ {percentile} pct)")
            print(f"{'='*55}\n")

        return result

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Randomises initial pitch angle/rate, altitude, heading, airspeed,
        regenerates the flight path and atmosphere, and applies any pending
        episode-length change. Returns (obs, info).
        """
        super().reset(seed=seed)

        self._apply_pending_max_steps()

        if self.turbulence_severity == "random":
            self._current_severity = self.np_random.choice(
                self._SEVERITIES, p=self._severity_weights
            )
            self._atmosphere.turbulence._apply_severity(self._current_severity)

        theta     = self.np_random.uniform(-np.radians(15), np.radians(15))
        theta_dot = self.np_random.uniform(-0.2, 0.2)
        self._state = np.array([theta, theta_dot], dtype=np.float64)

        self._steps       = 0
        self._time        = 0.0
        self._history     = [self._state.copy()]
        self._prev_torque = 0.0
        self._last_disturbance = 0.0
        self._theta_dot_pre = 0.0

        rng = np.random.default_rng(seed)
        self._flight_path.new_episode(rng)
        self._atmosphere.new_episode(rng)

        t0 = 0.0
        self.altitude = (self._flight_path.target_altitude(t0)
                         + self.np_random.uniform(-100, 100))
        self.heading  = (self._flight_path.target_heading(t0)
                         + self.np_random.uniform(-15, 15))
        self.airspeed = self.np_random.uniform(150, 300)
        self._prev_altitude = self.altitude

        return self._get_obs(), {}

    def step(self, action: int):
        """
        Advance the simulation by one timestep.

        Applies the selected torque, integrates pitch dynamics, updates altitude
        and heading, computes the reward signal, and checks termination conditions.

        Args:
            action: integer index into self._torques [0, N_ACTIONS).

        Returns:
            obs, reward, terminated, truncated, info dict.

        Reward components:
            -2.5 * tracking_error    primary pitch-tracking penalty
            -0.4 * rate_penalty      oscillation damping
            -0.5 * effort            discourages full-deflection inputs
            -0.3 * torque_delta      penalises rapid torque switching
            +severity_gain * stable_bonus   hard bonus inside STABLE_THRESH
            +severity_gain * soft_stable    Gaussian bonus centred on required_pitch
            alt_penalty / hdg_penalty       logarithmic outside tolerance
            divergence_penalty              discourages runaway pitch near limit
            cubic/quad angle penalty        strongly penalises large angles
            +severity_gain * 10.0           waypoint passage bonus
            -50 - 10*theta_dot_pre          fixed crash penalty
            +40 + severity_gain * 4.0       episode completion bonus
        """
        assert self.action_space.contains(action)
        u = self._torques[action]

        theta, theta_dot = self._state
        self._theta_dot_pre = float(theta_dot)
        prev_theta_dot = self._theta_dot_pre
        t = self._time

        dalt = self.altitude - self._prev_altitude
        atmo = self._atmosphere.step(
            altitude_m   = self.altitude,
            airspeed_mps = self.airspeed,
            heading_deg  = self.heading,
            dalt_m       = dalt,
            rng          = self.np_random,
        )
        density        = atmo["density"]
        disturbance    = atmo["total_torque"]
        gust_torque    = atmo["gust_torque"]
        shear_torque   = atmo["shear_torque"]
        thermal_torque = atmo["thermal_torque"]

        self._last_disturbance = disturbance

        target_alt = self._flight_path.target_altitude(t)
        target_hdg = self._flight_path.target_heading(t)
        target_roc = np.clip(
            self._flight_path.target_rate_of_climb(t),
            -self.MAX_ROC_MS, self.MAX_ROC_MS,
        )

        alt_error = self.altitude - target_alt
        hdg_error = _heading_error(self.heading, target_hdg)

        alt_pitch_bias = np.clip(
            -self.ALT_PITCH_COUPLING * alt_error,
            -self.ALT_PITCH_BIAS_MAX,
            self.ALT_PITCH_BIAS_MAX,
        )
        hdg_load = np.clip(
            self.HDG_ROLL_COUPLING * abs(hdg_error),
            0.0, self.HDG_ROLL_LOAD_MAX,
        )

        theta_ddot = (
            -self.D * density * theta_dot
            - self.K * theta
            + u
            + disturbance
            + alt_pitch_bias
        ) / self.I - hdg_load * np.sign(theta)

        theta_dot += theta_ddot * self.dt
        theta     += theta_dot  * self.dt
        self._state = np.array([theta, theta_dot], dtype=np.float64)

        self._prev_altitude = self.altitude
        climb_rate    = self.airspeed * np.sin(theta)
        self.altitude = np.clip(self.altitude + climb_rate * self.dt, 0, 12_000)

        heading_rate = -0.1 * hdg_error
        self.heading = (self.heading + heading_rate * self.dt) % 360.0

        self.airspeed = np.clip(
            self.airspeed + self.np_random.uniform(-1, 1), 100, 350
        )

        self._steps += 1
        self._time   = self._steps * self.dt
        self._history.append(self._state.copy())

        # Usa airspeed nominale fissa per required_pitch: evita che il rumore
        # random sull'airspeed (+/-1/step) renda il target instabile a parità
        # di ROC richiesto. 225 m/s è il centro del range [150,300].
        _nominal_airspeed = 225.0
        required_pitch = np.arcsin(
            np.clip(target_roc / _nominal_airspeed, -1.0, 1.0)
        )
        tracking_error = abs(theta - required_pitch)
        rate_penalty   = abs(theta_dot)
        stable_bonus = (
            1.0
            if (abs(theta - required_pitch) < self.STABLE_THRESH
                and abs(theta_dot) < 0.2)
            else 0.0
        )

        effort       = (u / self.T_MAX) ** 2
        torque_delta = abs(u - self._prev_torque) / (2 * self.T_MAX)
        self._prev_torque = u


        _alt_abs = abs(alt_error)
        if _alt_abs <= self.ALT_TOLERANCE_M:
            alt_penalty = -0.6 * (_alt_abs / self.ALT_TOLERANCE_M)
        else:
            alt_penalty = -0.6 - 0.5 * np.log(
                1.0 + (_alt_abs - self.ALT_TOLERANCE_M) / self.ALT_TOLERANCE_M
            )

        _hdg_abs = abs(hdg_error)
        if _hdg_abs <= self.HDG_TOLERANCE_DEG:
            hdg_penalty = -0.5 * (_hdg_abs / self.HDG_TOLERANCE_DEG)
        else:
            hdg_penalty = -0.5 - 0.4 * np.log(
                1.0 + (_hdg_abs - self.HDG_TOLERANCE_DEG) / self.HDG_TOLERANCE_DEG
            )

        tracking_error = np.clip(tracking_error, 0, 0.5) / 0.5
        rate_penalty   = np.clip(rate_penalty, 0, 1.0) / 1.0


        severity_gain = self.SEVERITY_REWARD_GAIN.get(self._current_severity, 1.0)

        tracking_w = 3.2
        rate_w     = 0.6
        effort_w   = 0.7
        torque_w   = 0.4
        stable_w   = 2.5

        reward = (
            - tracking_w * tracking_error
            - rate_w * rate_penalty
            - effort_w * effort
            - torque_w * torque_delta
            + severity_gain * stable_w * stable_bonus
            + alt_penalty
            + hdg_penalty
        )

        soft_stable = np.exp(-self.SOFT_STABLE_BONUS_DECAY * ((theta - required_pitch) / self.THETA_LIMIT) ** 2)
        soft_gain = 1.2 * severity_gain
        reward += soft_gain * soft_stable

       
        proximity = abs(theta) / self.THETA_LIMIT
        if proximity > self.DIVERGENCE_PROXIMITY_FRAC and np.sign(theta) == np.sign(theta_dot):
            div_gain = self.DIVERGENCE_PENALTY_GAIN * 1.5
            reward -= div_gain * proximity * abs(theta_dot)
        
        theta_abs   = abs(theta)
        threshold   = np.radians(self.ANGLE_PENALTY_THRESHOLD_DEG)
        cubic_start = np.radians(self.ANGLE_PENALTY_CUBIC_DEG)
        if theta_abs > cubic_start:
            excess = (theta_abs - cubic_start) / (self.THETA_LIMIT - cubic_start)
            reward -= 1.3 * self.ANGLE_PENALTY_GAIN_CUBIC * excess ** 3
        elif theta_abs > threshold:
            excess = (theta_abs - threshold) / (cubic_start - threshold)
            reward -= 1.2 * self.ANGLE_PENALTY_GAIN_QUAD * excess ** 2

        ttw = self._flight_path.time_to_next_waypoint(t)
        if 0 <= ttw < self.dt:
            if (abs(alt_error) < self.ALT_TOLERANCE_M
                    and abs(hdg_error) < self.HDG_TOLERANCE_DEG):
                reward += severity_gain * 10.0

        terminated = bool(
            abs(theta) > self.THETA_LIMIT
            or self.altitude <= 0
            or abs(theta_dot) > 8.0
        )
        truncated  = bool(self._steps >= self._effective_max_steps)

        if terminated:
            reward -= 50.0 + 10.0 * min(abs(self._theta_dot_pre), 5.0)

        elif truncated:
            # Bonus episodio completato: incoraggia la sopravvivenza
            reward += 40.0 + severity_gain * 4.0

        if self.render_mode == "human":
            self._render_human()

        return self._get_obs(), float(reward), terminated, truncated, {
            "theta_deg":      np.degrees(theta),
            "theta_dot":      theta_dot,
            "torque":         u,
            "stable":         stable_bonus > 0,
            "altitude":       self.altitude,
            "heading":        self.heading,
            "target_alt":     target_alt,
            "target_hdg":     target_hdg,
            "target_roc":     target_roc,
            "required_pitch": np.degrees(required_pitch),
            "alt_error":      alt_error,
            "hdg_error":      hdg_error,
            "gust_torque":    gust_torque,
            "shear_torque":   shear_torque,
            "thermal_torque": thermal_torque,
        }

    def render(self):
        """
        Render the current state. Returns an RGB array if render_mode="rgb_array",
        otherwise renders to screen (human mode) and returns None.
        """
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def close(self):
        """Close the matplotlib renderer if open."""
        if self._renderer is not None:
            import matplotlib.pyplot as plt
            plt.close(self._renderer[0])
            self._renderer = None

    def _get_obs(self) -> np.ndarray:
        """
        Build and return the 10-element normalised observation vector.
        See module docstring for feature definitions and ranges.
        """
        theta, theta_dot = self._state
        t = self._time

        target_alt  = self._flight_path.target_altitude(t)
        target_hdg  = self._flight_path.target_heading(t)
        target_roc  = np.clip(
            self._flight_path.target_rate_of_climb(t),
            -self.MAX_ROC_MS, self.MAX_ROC_MS,
        )
        atmo_density = self._atmosphere.isa.density(self.altitude)

        alt_error     = self.altitude - target_alt
        hdg_err       = _heading_error(self.heading, target_hdg)
        rate_of_climb = self.airspeed * np.sin(theta)
        roc_error     = rate_of_climb - target_roc
        ttw           = self._flight_path.time_to_next_waypoint(t)

        obs = np.array([
            np.clip(theta                 / self.THETA_LIMIT,  -1, 1),
            np.clip(theta_dot             / 5.0,               -1, 1),
            np.clip(alt_error             / 5_000.0,           -1, 1),
            np.clip(roc_error             / 20.0,              -1, 1),
            np.clip(hdg_err               / 180.0,             -1, 1),
            np.clip((self.airspeed - 225) / 75.0,              -1, 1),
            np.clip((atmo_density - 0.5)  / 0.5,               -1, 1),
            np.clip(ttw / max(self._ttw_norm, 1.0),             0,  1),
            np.clip(target_roc            / 20.0,              -1, 1),
            np.clip(self._last_disturbance / self.DISTURBANCE_OBS_SCALE,
                    -self.DISTURBANCE_OBS_CLIP, self.DISTURBANCE_OBS_CLIP),
        ], dtype=np.float32)
        return obs

    def _render_human(self):
        """
        Update the live matplotlib window with pitch, target altitude and target
        heading time-series. Creates the figure on the first call.
        matplotlib is imported lazily to avoid forcing it on headless workers.
        """
        import matplotlib.pyplot as plt
        if self._renderer is None:
            plt.ion()
            fig, axes = plt.subplots(3, 1, figsize=(9, 8))
            self._renderer = (fig, axes)
        fig, (ax1, ax2, ax3) = self._renderer
        for ax in (ax1, ax2, ax3):
            ax.clear()

        history = np.array(self._history)
        t_arr   = np.arange(len(history)) * self.dt
        t_max   = len(history) * self.dt
        t_plot  = np.linspace(0, t_max, len(history))

        ax1.plot(t_arr, np.degrees(history[:, 0]), "b", label=" theta")
        ax1.set_ylabel("Pitch [ [deg]]"); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(t_plot,
                 [self._flight_path.target_altitude(ti) for ti in t_plot],
                 "g--", lw=1, label="Target alt")
        ax2.set_ylabel("Altitude [m]"); ax2.legend(); ax2.grid(alpha=0.3)
        ax3.plot(t_plot,
                 [self._flight_path.target_heading(ti) for ti in t_plot],
                 "r--", lw=1, label="Target hdg")
        ax3.set_ylabel("Heading [ [deg]]"); ax3.legend(); ax3.grid(alpha=0.3)
        fig.tight_layout(); plt.pause(0.001)

    def _render_rgb_array(self):
        """
        Render pitch and altitude history to an off-screen PNG and return it
        as an HxWx3 uint8 numpy array.
        matplotlib is imported lazily to avoid forcing it on headless workers.
        """

        import matplotlib.pyplot as plt, io
        import PIL.Image
        fig, axes = plt.subplots(2, 1, figsize=(7, 5))
        history = np.array(self._history)
        t_arr   = np.arange(len(history)) * self.dt
        axes[0].plot(t_arr, np.degrees(history[:, 0]), "b", lw=1.5)
        axes[0].set_ylabel("Pitch [deg]"); axes[0].set_xlabel("Time [s]")
        t_max  = len(history) * self.dt
        t_plot = np.linspace(0, t_max, len(history))
        axes[1].plot(t_plot,
                     [self._flight_path.target_altitude(ti) for ti in t_plot],
                     "g--", lw=1, label="Target")
        axes[1].set_ylabel("Altitude [m]"); axes[1].legend()
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        buf.seek(0); return np.array(PIL.Image.open(buf))


def _heading_error(actual_deg: float, target_deg: float) -> float:
    """
    Signed heading error in degrees, wrapped to [-180, 180].
    Positive means actual is clockwise of target.
    """
    return float((actual_deg - target_deg + 180.0) % 360.0 - 180.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AircraftPitchEnv utilities (es. calibrazione T_MAX)"
    )
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibra T_MAX in base al disturbance reale.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--severity", type=str, default="severe",
                        choices=["light", "moderate", "severe"])
    parser.add_argument("--percentile", type=float, default=99.7)
    parser.add_argument("--probe-tmax", type=float, default=50.0)
    args = parser.parse_args()

    if args.calibrate:
        AircraftPitchEnv.calibrate_t_max(
            severity=args.severity,
            episodes=args.episodes,
            percentile=args.percentile,
            probe_tmax=args.probe_tmax,
        )
    else:
        parser.print_help()