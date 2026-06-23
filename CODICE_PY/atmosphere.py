"""
atmosphere.py
=============
Realistic atmospheric disturbance model combining four effects:
  1. ISA  - air density as a function of altitude
  2. DrydenTurbulence  - correlated Gaussian vertical gust (MIL-HDBK-1797)
  3. WindShear  - altitude-banded horizontal wind layers
  4. Thermals  - discrete rising/sinking air columns

All outputs are SI units unless noted.
"""

import numpy as np


class ISA:
    """
    International Standard Atmosphere up to 11 000 m.
    Provides static air density and temperature vs altitude.
    """
    T0    = 288.15
    P0    = 101325.0
    RHO0  = 1.225
    L     = 0.0065
    R     = 287.05
    G     = 9.80665

    @classmethod
    def density(cls, altitude_m: float) -> float:
        """Air density [kg/m³] at the given altitude."""
        alt = np.clip(altitude_m, 0.0, 11_000.0)
        T   = cls.T0 - cls.L * alt
        return cls.RHO0 * (T / cls.T0) ** (cls.G / (cls.R * cls.L) - 1.0)

    @classmethod
    def temperature(cls, altitude_m: float) -> float:
        """Static air temperature [K] at the given altitude."""
        return cls.T0 - cls.L * np.clip(altitude_m, 0.0, 11_000.0)


class DrydenTurbulence:
    """
    Scalar Dryden model for vertical gust velocity w_g.
    Implements a first-order autoregressive filter driven by white noise,
    parameterised by intensity (sigma_w) and length scale (L_w).
    Three named severity presets are available: light, moderate, severe.
    """

    # sigma_w and L_w values are intentionally conservative to remain
    # tractable without a difficulty curriculum. See atmosphere.py header.
    SEVERITY = {
        "light":    dict(sigma_w=0.5,  L_w=150.0),
        "moderate": dict(sigma_w=1.5,  L_w=80.0),
        "severe":   dict(sigma_w=2.0,  L_w=120.0),
    }

    def __init__(self, airspeed_mps: float = 75.0,
                 severity: str = "moderate",
                 dt: float = 0.05):
        self._airspeed = airspeed_mps
        self._dt       = dt
        self._w_g      = 0.0
        cfg = self.SEVERITY[severity]
        self._sigma_w = cfg["sigma_w"]
        self._L_w     = cfg["L_w"]

    def _apply_severity(self, severity: str) -> None:
        """Update sigma and length-scale parameters in-place."""
        cfg = self.SEVERITY[severity]
        self._sigma_w = cfg["sigma_w"]
        self._L_w     = cfg["L_w"]

    def update(self, airspeed_mps: float = None,
               severity: str = None,
               rng: np.random.Generator = None) -> float:
        """
        Advance the turbulence filter by one timestep.
        Returns the new vertical gust velocity [m/s].
        Optionally refreshes airspeed or severity before stepping.
        """
        if airspeed_mps is not None:
            self._airspeed = airspeed_mps
        if severity is not None:
            self._apply_severity(severity)
        rng  = rng or np.random.default_rng()
        V    = max(self._airspeed, 10.0)
        tau  = self._L_w / V
        a    = np.exp(-self._dt / tau)
        b    = self._sigma_w * np.sqrt(1 - a**2)
        self._w_g = a * self._w_g + b * rng.standard_normal()
        return self._w_g

    def reset(self):
        """Reset the gust state to zero (call at the start of each episode)."""
        self._w_g = 0.0


class WindShear:
    """
    Altitude-banded horizontal wind model.
    Each episode generates three random band boundaries and assigns
    independent (wind_u, wind_v) vectors to each band.
    """

    def __init__(self, rng: np.random.Generator = None):
        self._rng    = rng or np.random.default_rng()
        self._layers = []
        self.new_episode()

    def new_episode(self, rng: np.random.Generator = None):
        """Regenerate random wind layers for a new episode."""
        rng = rng or self._rng
        band_tops = np.sort(rng.uniform(1_000, 10_000, size=3))
        self._layers = []
        prev_top = 0.0
        for top in np.append(band_tops, 12_000):
            self._layers.append({
                "bottom": prev_top,
                "top":    top,
                "wind_u": rng.uniform(-15, 15),
                "wind_v": rng.uniform(-10, 10),
            })
            prev_top = top

    def wind_at(self, altitude_m: float) -> tuple:
        """Return the (wind_u, wind_v) [m/s] of the layer containing altitude_m."""
        for layer in self._layers:
            if layer["bottom"] <= altitude_m < layer["top"]:
                return layer["wind_u"], layer["wind_v"]
        return self._layers[-1]["wind_u"], self._layers[-1]["wind_v"]

    def shear_force(self, altitude_m: float, dalt_m: float) -> float:
        """
        Approximate pitch-axis shear torque when crossing a wind boundary.
        Returns 0 if the altitude change is negligible.
        """
        if abs(dalt_m) < 0.1:
            return 0.0
        u0, _ = self.wind_at(altitude_m)
        u1, _ = self.wind_at(altitude_m + dalt_m)
        delta_u = u1 - u0
        return 0.3 * delta_u


class Thermals:
    """
    Discrete rising/sinking air columns distributed over a 20x20 km grid.
    Half the columns are updrafts (positive strength) and half are downdrafts
    (negative), preventing a systematic upward bias on pitch disturbance.
    The aircraft position is tracked on a toroidal grid and lift is computed
    via a Gaussian proximity kernel scaled by altitude.
    """

    def __init__(self, n_thermals: int = 4, rng: np.random.Generator = None):
        self._n   = n_thermals
        self._rng = rng or np.random.default_rng()
        self.new_episode()

    def new_episode(self, rng: np.random.Generator = None):
        """Generate a new random set of thermal columns for an episode."""
        rng = rng or self._rng
        n_up   = self._n // 2
        n_down = self._n - n_up
        if self._n % 2 == 1 and rng.random() < 0.5:
            n_up, n_down = n_down, n_up
        signs = np.array([1.0] * n_up + [-1.0] * n_down)
        rng.shuffle(signs)
        self._thermals = [
            {
                "x":        rng.uniform(0, 20),
                "y":        rng.uniform(0, 20),
                "radius":   rng.uniform(1, 4),
                "strength": signs[i] * rng.uniform(0.5, 3.0),
                "max_alt":  rng.uniform(1_000, 6_000),
            }
            for i in range(self._n)
        ]
        self._pos_x = rng.uniform(0, 20)
        self._pos_y = rng.uniform(0, 20)

    def step(self, heading_deg: float, airspeed_mps: float,
             altitude_m: float, dt: float) -> float:
        """
        Advance the aircraft position on the thermal grid and return the
        total vertical lift torque [N*m] from all nearby thermals.
        """
        hdg_rad  = np.radians(heading_deg)
        spd_norm = airspeed_mps / 75.0
        self._pos_x += spd_norm * np.sin(hdg_rad) * dt
        self._pos_y += spd_norm * np.cos(hdg_rad) * dt
        self._pos_x %= 20.0
        self._pos_y %= 20.0
        total_lift = 0.0
        for t in self._thermals:
            dist = np.hypot(self._pos_x - t["x"], self._pos_y - t["y"])
            if dist < t["radius"] and altitude_m < t["max_alt"]:
                strength   = t["strength"] * np.exp(-0.5 * (dist / t["radius"])**2)
                alt_factor = 1.0 - altitude_m / t["max_alt"]
                total_lift += strength * alt_factor
        return 0.8 * total_lift


class AtmosphereModel:
    """
    Aggregates ISA, DrydenTurbulence, WindShear and Thermals into a single
    step() call that returns a dict of per-component and total torques plus
    current air density.
    """

    def __init__(self, dt: float = 0.05,
                 turbulence_severity: str = "moderate",
                 rng: np.random.Generator = None):
        self._rng       = rng or np.random.default_rng()
        self.turbulence = DrydenTurbulence(dt=dt, severity=turbulence_severity)
        self.shear      = WindShear(rng=self._rng)
        self.thermals   = Thermals(rng=self._rng)
        self.isa        = ISA()

    def new_episode(self, rng: np.random.Generator = None):
        """Reset all sub-models for a new episode."""
        rng = rng or self._rng
        self.turbulence.reset()
        self.shear.new_episode(rng)
        self.thermals.new_episode(rng)

    def step(self, altitude_m: float, airspeed_mps: float,
             heading_deg: float, dalt_m: float,
             rng: np.random.Generator) -> dict:
        """
        Advance all disturbance models by one timestep and return:
            density        - ISA air density [kg/m³]
            gust_torque    - Dryden vertical gust contribution [N*m]
            shear_torque   - wind-shear contribution [N*m]
            thermal_torque - thermal lift/sink contribution [N*m]
            total_torque   - sum of the three torque components
        """
        density        = self.isa.density(altitude_m)
        gust_torque    = self.turbulence.update(airspeed_mps, rng=rng)
        shear_torque   = self.shear.shear_force(altitude_m, dalt_m)
        thermal_torque = self.thermals.step(heading_deg, airspeed_mps,
                                            altitude_m, self.turbulence._dt)
        return {
            "density":        density,
            "gust_torque":    gust_torque,
            "shear_torque":   shear_torque,
            "thermal_torque": thermal_torque,
            "total_torque":   gust_torque + shear_torque + thermal_torque,
        }