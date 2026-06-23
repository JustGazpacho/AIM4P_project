"""
flight_path.py
==============
Physically-plausible 3-D flight path generator.
Produces smooth spline routes (altitude + heading vs time) that an aircraft
must track during a training or evaluation episode.

Design constraints
------------------
- Episode duration is passed in from the environment (default 120 s).
- An initial cruise phase (first 15 % of episode, min 10 s) holds altitude
  and heading constant so the agent can stabilise before the first manoeuvre.
- Altitude steps between waypoints are sized so the required rate-of-climb
  never exceeds 30 % of MAX_ROC_MS at any spline knot, leaving headroom for
  Akima over-shoots and atmospheric disturbances.
- Heading changes are limited to 30 [deg] per waypoint segment.
- Altitude is constrained to [500, 10 000] m.

Coordinate convention
---------------------
    altitude : metres above sea level  (0-12 000 m)
    heading  : degrees from north      (0-360 [deg], unwrapped for spline)
    time     : seconds from episode start
"""

import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline


class FlightPath:
    """
    Time-parameterised flight path built from random altitude and heading
    waypoints.  Altitude uses an Akima spline (no overshoot between knots);
    heading uses a cubic spline on unwrapped degree values.
    """

    STEP_ROC_BUDGET:  float = 0.30   # fraction of MAX_ROC_MS used to size altitude steps
    CRUISE_FRACTION:  float = 0.15   # initial fraction of episode held as straight-level cruise
    MIN_CRUISE_S:     float = 10.0   # minimum cruise phase [s]
    MAX_HEADING_STEP: float = 30.0   # maximum heading change per waypoint segment [deg]

    def __init__(
        self,
        n_waypoints: int   = 6,
        alt_range:   tuple = (500, 10_000),
        duration:    float = 120.0,
        rng:         np.random.Generator = None,
    ) -> None:
        """
        Args:
            n_waypoints: number of manoeuvre waypoints after the cruise phase.
            alt_range:   (min, max) altitude [m] for waypoint sampling.
            duration:    total episode duration [s].
            rng:         optional random generator; a fresh one is created if None.
        """
        from aircraft_pitch_env import AircraftPitchEnv as _Env
        self.MAX_ROC_MS = _Env.MAX_ROC_MS

        self.n_waypoints = n_waypoints
        self.alt_range   = alt_range
        self.duration    = duration
        self._rng        = rng or np.random.default_rng()

        self._alt_spline:     Akima1DInterpolator | None = None
        self._hdg_spline:     CubicSpline          | None = None
        self._waypoint_times: np.ndarray            | None = None
        self._waypoints_alt:  np.ndarray            | None = None
        self._waypoints_hdg:  np.ndarray            | None = None

        self.roc_clamp_count: int = 0

        self.new_episode()

    def new_episode(self, rng: np.random.Generator = None) -> None:
        """Generate a fresh random path for one episode."""
        rng = rng or self._rng

        cruise_duration = max(self.MIN_CRUISE_S,
                              self.CRUISE_FRACTION * self.duration)
        manoeuvre_times = np.linspace(cruise_duration, self.duration, self.n_waypoints)
        times = np.concatenate([[0.0], manoeuvre_times])
        seg_duration = manoeuvre_times[1] - manoeuvre_times[0]

        max_alt_step = self.MAX_ROC_MS * seg_duration * 0.30
        max_alt_step = min(max_alt_step, 150.0)

        start_alt = float(rng.uniform(*self.alt_range))
        alts_manoeuvre = self._random_walk(rng, *self.alt_range, max_alt_step, start=start_alt)
        alts = np.concatenate([[start_alt], alts_manoeuvre])

        start_hdg = float(rng.uniform(0.0, 360.0))
        hdgs_manoeuvre = self._random_walk_heading(rng, self.MAX_HEADING_STEP, start=start_hdg)
        hdgs = np.concatenate([[start_hdg], hdgs_manoeuvre])

        self._alt_spline = Akima1DInterpolator(times, alts)
        hdgs = np.rad2deg(np.unwrap(np.deg2rad(hdgs)))
        self._hdg_spline = CubicSpline(times, hdgs)

        self._waypoint_times = times
        self._waypoints_alt  = alts
        self._waypoints_hdg  = hdgs % 360.0
        self.roc_clamp_count = 0

    def target_altitude(self, t: float) -> float:
        """Desired altitude [m] at time t [s]."""
        return float(self._alt_spline(np.clip(t, 0.0, self.duration)))

    def target_heading(self, t: float) -> float:
        """Desired heading [deg, 0-360) at time t [s]."""
        return float(self._hdg_spline(np.clip(t, 0.0, self.duration)) % 360.0)

    def target_rate_of_climb(self, t: float) -> float:
        """
        Desired rate of climb [m/s] at time t, approximated by a centred
        finite difference of the altitude spline.  Clamped to +/-MAX_ROC_MS.
        """
        t   = np.clip(t, 0.0, self.duration)
        eps = min(0.1, self.duration * 0.001)
        t_lo = max(t - eps, 0.0)
        t_hi = min(t + eps, self.duration)
        raw = float((self._alt_spline(t_hi) - self._alt_spline(t_lo)) / (t_hi - t_lo))
        clamped = float(np.clip(raw, -self.MAX_ROC_MS, self.MAX_ROC_MS))
        if clamped != raw:
            self.roc_clamp_count += 1
        return clamped

    def time_to_next_waypoint(self, t: float) -> float:
        """Seconds until the next waypoint; used as an observation feature."""
        future = self._waypoint_times[self._waypoint_times > t]
        return float(future[0] - t) if len(future) > 0 else 0.0

    @property
    def waypoints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(times, altitudes, headings) arrays; useful for visualisation."""
        return self._waypoint_times, self._waypoints_alt, self._waypoints_hdg

    def _random_walk(
        self,
        rng:      np.random.Generator,
        lo:       float,
        hi:       float,
        max_step: float,
        start:    float = None,
    ) -> np.ndarray:
        """
        Bounded random walk of length n_waypoints.
        Each step is drawn from Uniform[-max_step, +max_step] and the
        running value is clipped to [lo, hi].
        """
        v = float(rng.uniform(lo, hi)) if start is None else start
        values = []
        for _ in range(self.n_waypoints):
            v = float(np.clip(v + rng.uniform(-max_step, max_step), lo, hi))
            values.append(v)
        return np.array(values)

    def _random_walk_heading(
        self,
        rng:      np.random.Generator,
        max_step: float,
        start:    float = None,
    ) -> np.ndarray:
        """
        Unbounded random walk for heading (unwrapped) so the cubic spline
        interpolates through smooth turns rather than 360 [deg]-wrap artefacts.
        """
        v = float(rng.uniform(0.0, 360.0)) if start is None else start
        values = []
        for _ in range(self.n_waypoints):
            v += rng.uniform(-max_step, max_step)
            values.append(v)
        return np.array(values)