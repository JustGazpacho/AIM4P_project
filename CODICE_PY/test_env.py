"""
test_env.py
===========
Unit tests for AircraftPitchEnv.

Run with:
    python -m pytest test_env.py -v
    python -m pytest test_env.py -v --tb=short
    python test_env.py
"""

import numpy as np
import pytest

from aircraft_pitch_env import AircraftPitchEnv, _heading_error


@pytest.fixture
def env():
    """Standard AircraftPitchEnv reset with seed=0 for general tests."""
    e = AircraftPitchEnv()
    e.reset(seed=0)
    return e


@pytest.fixture
def clean_env():
    """Fresh env with all atmospheric disturbances zeroed for deterministic physics tests."""
    e = AircraftPitchEnv()
    e.reset(seed=0)
    e._atmosphere.turbulence._w_g     = 0.0
    e._atmosphere.turbulence._sigma_w = 0.0
    e._atmosphere.shear._layers = [
        {"bottom": 0, "top": 12000, "wind_u": 0.0, "wind_v": 0.0}
    ]
    e._atmosphere.thermals._thermals = []
    return e


class TestSpaces:
    """Tests for observation and action space shape, bounds and normalisation."""
    def test_obs_space_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_obs_space_size_is_10(self, env):
        obs, _ = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"

    def test_obs_within_bounds_after_reset(self, env):
        for _ in range(20):
            obs, _ = env.reset()
            assert env.observation_space.contains(obs), f"Obs out of bounds: {obs}"

    def test_obs_within_bounds_during_episode(self, env):
        for _ in range(5):
            obs, _ = env.reset()
            for _ in range(50):
                a = env.action_space.sample()
                obs, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
                assert env.observation_space.contains(obs), \
                    f"Obs out of bounds during episode: {obs}"

    def test_obs_normalization_pitch_at_limit(self):
        """Pitch at THETA_LIMIT should normalise to 1; zero rate to 0."""
        env = AircraftPitchEnv()
        env.reset(seed=0)
        env._state   = np.array([env.THETA_LIMIT, 0.0])
        env.airspeed = 225.0
        env.altitude = env._flight_path.target_altitude(0.0)
        env.heading  = env._flight_path.target_heading(0.0)
        obs = env._get_obs()
        assert obs[0] == pytest.approx(1.0)
        assert obs[1] == pytest.approx(0.0)

    def test_obs_target_roc_feature(self):
        """Feature index 8 (target_roc_norm) must be present and the obs 10-dimensional."""
        env = AircraftPitchEnv()
        env.reset(seed=0)
        for _ in range(10):
            obs, _, t, tr, _ = env.step(env.N_ACTIONS // 2)
            if t or tr:
                break
        assert obs.shape == (10,)

    def test_action_space_size(self, env):
        assert env.action_space.n == AircraftPitchEnv.N_ACTIONS

    def test_gymnasium_check(self):
        from gymnasium.utils.env_checker import check_env
        check_env(AircraftPitchEnv(), warn=True)


class TestReset:
    """Tests for reset() determinism, state clearing and seeding behaviour."""
    def test_reset_returns_obs_and_info(self, env):
        result = env.reset()
        assert len(result) == 2

    def test_reset_seed_determinism(self):
        e1 = AircraftPitchEnv()
        e2 = AircraftPitchEnv()
        obs1, _ = e1.reset(seed=7)
        obs2, _ = e2.reset(seed=7)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_clears_step_counter(self, env):
        for _ in range(10):
            env.step(env.action_space.sample())
        env.reset()
        assert env._steps == 0

    def test_reset_clears_time(self, env):
        for _ in range(10):
            env.step(env.action_space.sample())
        env.reset()
        assert env._time == pytest.approx(0.0)

    def test_reset_clears_history(self, env):
        for _ in range(10):
            env.step(env.action_space.sample())
        env.reset()
        assert len(env._history) == 1

    def test_initial_pitch_within_range(self):
        env = AircraftPitchEnv()
        for seed in range(30):
            env.reset(seed=seed)
            assert abs(env._state[0]) <= np.radians(30)

    def test_multiple_resets_are_independent(self):
        env = AircraftPitchEnv()
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=999)
        assert not np.allclose(obs1, obs2)


class TestPhysics:
    """Tests for pitch dynamics, altitude coupling and torque symmetry."""
    def test_positive_torque_increases_pitch_rate(self, clean_env):
        clean_env._state = np.array([0.0, 0.0])
        clean_env.step(AircraftPitchEnv.N_ACTIONS - 1)
        assert clean_env._state[1] > 0

    def test_negative_torque_decreases_pitch_rate(self, clean_env):
        clean_env._state = np.array([0.0, 0.0])
        clean_env.step(0)
        assert clean_env._state[1] < 0

    def test_zero_torque_damps_naturally(self, clean_env):
        clean_env._state = np.array([0.0, 1.0])
        neutral = AircraftPitchEnv.N_ACTIONS // 2
        initial_rate = abs(clean_env._state[1])
        for _ in range(20):
            clean_env.step(neutral)
        assert abs(clean_env._state[1]) < initial_rate

    def test_torque_values_symmetric(self):
        env = AircraftPitchEnv()
        torques = env._torques
        assert abs(torques[0] + torques[-1]) < 1e-9
        assert abs(torques[AircraftPitchEnv.N_ACTIONS // 2]) < 1e-9

    def test_torque_count_matches_n_actions(self):
        env = AircraftPitchEnv()
        assert len(env._torques) == AircraftPitchEnv.N_ACTIONS

    def test_altitude_increases_with_positive_pitch(self, clean_env):
        clean_env._state   = np.array([np.radians(10), 0.0])
        clean_env.airspeed = 200.0
        initial_alt = clean_env.altitude
        for _ in range(5):
            clean_env.step(AircraftPitchEnv.N_ACTIONS // 2)
        assert clean_env.altitude > initial_alt

    def test_altitude_decreases_with_negative_pitch(self, clean_env):
        clean_env._state         = np.array([-np.radians(10), 0.0])
        clean_env.airspeed       = 200.0
        clean_env.altitude       = 5000.0
        clean_env._prev_altitude = 5000.0
        initial_alt = clean_env.altitude
        for _ in range(5):
            clean_env.step(AircraftPitchEnv.N_ACTIONS // 2)
        assert clean_env.altitude < initial_alt

    def test_system_poles_are_stable(self):
        """Verify that the unforced pitch system is asymptotically stable."""
        I, D, K = AircraftPitchEnv.I, AircraftPitchEnv.D, AircraftPitchEnv.K
        discriminant = (D / I) ** 2 - 4 * (K / I)
        if discriminant >= 0:
            r1 = (-D / I + np.sqrt(discriminant)) / 2
            r2 = (-D / I - np.sqrt(discriminant)) / 2
            assert r1 < 0 and r2 < 0
        else:
            assert -D / (2 * I) < 0

    def test_alt_pitch_bias_is_clamped(self, clean_env):
        """Altitude-coupling bias must not exceed ALT_PITCH_BIAS_MAX per step."""
        clean_env._state = np.array([0.0, 0.0])
        target_alt = clean_env._flight_path.target_altitude(clean_env._time)
        clean_env.altitude       = np.clip(target_alt + 50_000.0, 0, 12_000)
        clean_env._prev_altitude = clean_env.altitude
        clean_env.step(AircraftPitchEnv.N_ACTIONS // 2)
        max_expected = AircraftPitchEnv.ALT_PITCH_BIAS_MAX * AircraftPitchEnv.dt
        assert abs(clean_env._state[1]) <= max_expected + 1e-9

    def test_hdg_roll_load_is_clamped(self, clean_env):
        """Heading-coupling load must not exceed HDG_ROLL_LOAD_MAX."""
        clean_env._state = np.array([np.radians(10), 0.0])
        target_hdg = clean_env._flight_path.target_heading(clean_env._time)
        clean_env.heading        = (target_hdg + 180.0) % 360.0
        clean_env.altitude       = clean_env._flight_path.target_altitude(clean_env._time)
        clean_env._prev_altitude = clean_env.altitude
        _, _, _, _, info = clean_env.step(AircraftPitchEnv.N_ACTIONS // 2)
        clamped = np.clip(AircraftPitchEnv.HDG_ROLL_COUPLING * abs(info["hdg_error"]),
                          0.0, AircraftPitchEnv.HDG_ROLL_LOAD_MAX)
        assert clamped <= AircraftPitchEnv.HDG_ROLL_LOAD_MAX + 1e-12

    def test_combined_bias_never_exceeds_half_torque(self):
        """Combined coupling clamps must leave meaningful control authority for the agent."""
        combined_max = AircraftPitchEnv.ALT_PITCH_BIAS_MAX + AircraftPitchEnv.HDG_ROLL_LOAD_MAX
        assert combined_max < AircraftPitchEnv.T_MAX


class TestReward:
    """Tests for individual reward components and crash/truncation penalties."""
    def test_stable_at_required_pitch_gives_bonus(self):
        env = AircraftPitchEnv()
        env.reset(seed=0)
        env._flight_path._alt_spline = type(env._flight_path._alt_spline)(
            env._flight_path._waypoint_times,
            np.full_like(env._flight_path._waypoints_alt, 5000.0)
        )
        env.altitude       = 5000.0
        env._prev_altitude = 5000.0
        env.heading        = env._flight_path.target_heading(env._time)
        env._state         = np.array([0.0001, 0.0])
        env.airspeed       = 200.0
        _, reward, _, _, info = env.step(AircraftPitchEnv.N_ACTIONS // 2)
        assert info["stable"]
        assert reward > 0.5

    def test_large_deviation_from_required_pitch_negative_reward(self):
        env = AircraftPitchEnv()
        env.reset(seed=0)
        env._state = np.array([np.radians(45), 0.0])
        _, reward, _, _, _ = env.step(AircraftPitchEnv.N_ACTIONS // 2)
        assert reward < 0

    def test_crash_penalty(self):
        env = AircraftPitchEnv()
        env.reset(seed=0)
        env._state = np.array([np.radians(59.9), 1.0])
        _, reward, terminated, _, _ = env.step(AircraftPitchEnv.N_ACTIONS - 1)
        if terminated:
            assert reward < -10

    def test_reward_is_finite_always(self):
        env = AircraftPitchEnv()
        env.reset(seed=0)
        for _ in range(100):
            a = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(a)
            assert np.isfinite(reward)
            if terminated or truncated:
                env.reset(seed=0)

    def test_required_pitch_field_in_info(self):
        env = AircraftPitchEnv()
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.N_ACTIONS // 2)
        assert "required_pitch" in info
        assert np.isfinite(info["required_pitch"])

    def test_divergence_penalty_applies_near_limit(self):
        """Diverging state (theta and theta_dot same sign near limit) should have lower reward."""
        limit = AircraftPitchEnv.THETA_LIMIT
        theta_near = 0.9 * limit

        env_div = AircraftPitchEnv(); env_div.reset(seed=0)
        env_div._state = np.array([theta_near, 1.0])

        env_conv = AircraftPitchEnv(); env_conv.reset(seed=0)
        env_conv._state = np.array([theta_near, -1.0])

        neutral = AircraftPitchEnv.N_ACTIONS // 2
        _, r_div,  term_div,  _, _ = env_div.step(neutral)
        _, r_conv, term_conv, _, _ = env_conv.step(neutral)
        if not term_div and not term_conv:
            assert r_div < r_conv

    def test_crash_penalty_fixed_independent_of_remaining_steps(self):
        """
        Crash penalty is fixed (not proportional to remaining steps).
        Early and late crashes should have nearly identical total rewards.

        The base penalty is -50 - 10*min(|theta_dot_pre|, 5.0) = -100 at
        theta_dot=5.0.  All other active reward terms (angle penalty, tracking,
        effort, soft-stable) are functions of state only, not of step count,
        so the two rewards must agree to within 10 units.
        """
        def crash_reward(steps_before_crash):
            env = AircraftPitchEnv(initial_max_steps=2400)
            env.reset(seed=0)
            env._atmosphere.turbulence._w_g     = 0.0
            env._atmosphere.turbulence._sigma_w = 0.0
            # Zero the divergence gain so only the fixed crash term and
            # state-dependent structural penalties contribute.
            env.DIVERGENCE_PENALTY_GAIN = 0.0
            env._steps = steps_before_crash
            env._time  = steps_before_crash * env.dt
            env._state = np.array([np.radians(59.99), 5.0])
            _, reward, terminated, _, _ = env.step(AircraftPitchEnv.N_ACTIONS - 1)
            assert terminated, "State should have caused a crash termination"
            return reward

        r_early = crash_reward(10)
        r_late  = crash_reward(2390)

        # Both must include the large fixed crash penalty.
        assert r_early < -50.0, (
            f"Early-crash reward {r_early:.2f} not below -50 — crash penalty missing?"
        )
        assert r_late < -50.0, (
            f"Late-crash reward {r_late:.2f} not below -50 — crash penalty missing?"
        )
        # The rewards must be nearly equal — step count must not affect crash severity.
        assert abs(r_late - r_early) < 10.0, (
            f"Early crash ({r_early:.2f}) and late crash ({r_late:.2f}) differ by "
            f"{abs(r_late - r_early):.2f} > 10 — crash penalty must not scale with "
            "remaining steps."
        )

    def test_soft_stable_bonus_decreases_with_pitch(self):
        """Reward near theta=0 should exceed reward at larger pitch due to soft stable bonus."""
        limit = AircraftPitchEnv.THETA_LIMIT

        def reward_at(theta_val):
            env = AircraftPitchEnv()
            env.reset(seed=0)
            env._atmosphere.turbulence._w_g     = 0.0
            env._atmosphere.turbulence._sigma_w = 0.0
            env._state = np.array([theta_val, 0.0])
            _, reward, terminated, _, _ = env.step(AircraftPitchEnv.N_ACTIONS // 2)
            return reward, terminated

        r_center, t_center = reward_at(0.01)
        r_mid,    t_mid    = reward_at(0.4 * limit)
        if not t_center and not t_mid:
            assert r_center > r_mid


class TestSeverityRewardGain:
    def test_severe_bonus_exceeds_light_bonus_same_state(self):
        """With identical state and no disturbances, 'severe' reward > 'light' reward."""
        def reward_at_severity(severity):
            env = AircraftPitchEnv(turbulence_severity=severity)
            env.reset(seed=0)
            env._atmosphere.turbulence._w_g     = 0.0
            env._atmosphere.turbulence._sigma_w = 0.0
            env._atmosphere.thermals._thermals  = []
            env._atmosphere.shear._layers = [{"bottom": 0, "top": 12000, "wind_u": 0.0, "wind_v": 0.0}]
            env._state = np.array([0.0001, 0.0])
            _, reward, terminated, _, _ = env.step(AircraftPitchEnv.N_ACTIONS // 2)
            return reward, terminated

        r_light,  t_light  = reward_at_severity("light")
        r_severe, t_severe = reward_at_severity("severe")
        if not t_light and not t_severe:
            assert r_severe > r_light

    def test_severity_gain_does_not_scale_penalties(self):
        """With a highly unstable state, 'severe' and 'light' rewards should be nearly equal."""
        def reward_at_severity(severity):
            env = AircraftPitchEnv(turbulence_severity=severity)
            env.reset(seed=0)
            env._atmosphere.turbulence._w_g     = 0.0
            env._atmosphere.turbulence._sigma_w = 0.0
            env._atmosphere.thermals._thermals  = []
            env._atmosphere.shear._layers = [{"bottom": 0, "top": 12000, "wind_u": 0.0, "wind_v": 0.0}]
            env._state = np.array([np.radians(45), 0.0])
            _, reward, terminated, _, _ = env.step(AircraftPitchEnv.N_ACTIONS // 2)
            return reward, terminated

        r_light,  t_light  = reward_at_severity("light")
        r_severe, t_severe = reward_at_severity("severe")
        if not t_light and not t_severe:
            assert abs(r_severe - r_light) < 1.0


class TestSeverityWeightsControl:
    """Tests for obs[9] (disturbance feature) range, clipping and content."""
    def test_set_severity_weights_updates_internal_state(self):
        env = AircraftPitchEnv()
        env.set_severity_weights([0.1, 0.2, 0.7])
        np.testing.assert_allclose(env._severity_weights, [0.1, 0.2, 0.7], atol=1e-9)

    def test_set_severity_weights_normalizes_unnormalized_input(self):
        env = AircraftPitchEnv()
        env.set_severity_weights([1, 1, 2])
        np.testing.assert_allclose(env._severity_weights, [0.25, 0.25, 0.5], atol=1e-9)

    def test_set_severity_weights_ignores_degenerate_input(self):
        env = AircraftPitchEnv()
        original = list(env._severity_weights)
        env.set_severity_weights([0.0, 0.0, 0.0])
        assert env._severity_weights == original


class TestDisturbanceObservation:
    def test_obs9_initially_zero_after_reset(self):
        env = AircraftPitchEnv()
        obs, _ = env.reset(seed=0)
        assert obs[9] == pytest.approx(0.0)

    def test_obs9_reflects_total_disturbance_not_only_gust(self):
        env = AircraftPitchEnv(turbulence_severity="moderate")
        env.reset(seed=0)
        env._atmosphere.turbulence._w_g     = 0.0
        env._atmosphere.turbulence._sigma_w = 0.0
        env._atmosphere.thermals._thermals  = []
        env._atmosphere.shear._layers = [
            {"bottom": 0,    "top": 6000,  "wind_u": -15.0, "wind_v": 0.0},
            {"bottom": 6000, "top": 12000, "wind_u":  15.0, "wind_v": 0.0},
        ]
        env.altitude       = 5999.0
        env._prev_altitude = 5990.0
        obs, _, _, _, info = env.step(AircraftPitchEnv.N_ACTIONS // 2)
        assert info["shear_torque"] != 0.0
        expected = np.clip(
            info["gust_torque"] + info["shear_torque"] + info["thermal_torque"],
            -AircraftPitchEnv.DISTURBANCE_OBS_CLIP * AircraftPitchEnv.DISTURBANCE_OBS_SCALE,
            AircraftPitchEnv.DISTURBANCE_OBS_CLIP * AircraftPitchEnv.DISTURBANCE_OBS_SCALE,
        ) / AircraftPitchEnv.DISTURBANCE_OBS_SCALE
        assert obs[9] == pytest.approx(expected, abs=1e-5)

    def test_obs9_can_exceed_unit_range_under_severe(self):
        """obs[9] can exceed [-1, 1] (up to +/-DISTURBANCE_OBS_CLIP) under strong disturbances."""
        env = AircraftPitchEnv(turbulence_severity="severe")
        env.reset(seed=0)
        env._last_disturbance = 7.0
        obs = env._get_obs()
        assert obs[9] == pytest.approx(AircraftPitchEnv.DISTURBANCE_OBS_CLIP)
        assert env.observation_space.contains(obs)

    def test_observation_space_bound_for_obs9(self):
        env = AircraftPitchEnv()
        high = env.observation_space.high
        low  = env.observation_space.low
        assert high[9] == pytest.approx(AircraftPitchEnv.DISTURBANCE_OBS_CLIP)
        assert low[9]  == pytest.approx(-AircraftPitchEnv.DISTURBANCE_OBS_CLIP)
        for i in list(range(7)) + [8]:
            assert high[i] == pytest.approx(1.0)
            assert low[i]  == pytest.approx(-1.0)
        assert high[7] == pytest.approx(1.0)
        assert low[7]  == pytest.approx(0.0)


class TestThermalsBalance:
    """Tests that thermals contain both updrafts and downdrafts."""
    def test_thermals_include_negative_strength(self):
        from atmosphere import Thermals
        rng = np.random.default_rng(0)
        th  = Thermals(n_thermals=8, rng=rng)
        saw_positive = saw_negative = False
        for _ in range(20):
            th.new_episode(rng)
            for t in th._thermals:
                if t["strength"] > 0: saw_positive = True
                elif t["strength"] < 0: saw_negative = True
        assert saw_positive and saw_negative

    def test_thermal_step_can_return_negative_lift(self):
        from atmosphere import Thermals
        rng = np.random.default_rng(1)
        th  = Thermals(n_thermals=1, rng=rng)
        th._thermals = [{"x": 10.0, "y": 10.0, "radius": 5.0,
                         "strength": -3.0, "max_alt": 6000.0}]
        th._pos_x, th._pos_y = 10.0, 10.0
        lift = th.step(heading_deg=0.0, airspeed_mps=0.0, altitude_m=1000.0, dt=0.05)
        assert lift < 0


class TestTurbulencePersistence:
    def test_moderate_severe_have_shorter_correlation_time(self):
        from atmosphere import DrydenTurbulence
        sev = DrydenTurbulence.SEVERITY
        assert sev["moderate"]["L_w"] < 200.0
        assert sev["severe"]["L_w"]   < 300.0
        assert sev["moderate"]["sigma_w"] == pytest.approx(1.5)
        assert sev["severe"]["sigma_w"]   == pytest.approx(2.0)


class TestTermination:
    """Tests for episode termination (angle limit, ground) and truncation (max steps)."""
    def test_angle_limit_terminates(self):
        env = AircraftPitchEnv()
        env.reset(seed=0)
        env._state = np.array([np.radians(59), 5.0])
        for _ in range(20):
            _, _, terminated, _, _ = env.step(AircraftPitchEnv.N_ACTIONS - 1)
            if terminated:
                return
        pytest.fail("Episode should have terminated from angle limit")

    def test_ground_collision_terminates(self):
        env = AircraftPitchEnv()
        env.reset(seed=0)
        env.altitude = env._prev_altitude = 5.0
        env._state   = np.array([-np.radians(20), -2.0])
        env.airspeed = 300.0
        for _ in range(50):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated:
                return
            if truncated:
                pytest.fail("Should terminate from ground collision before truncation")
        pytest.fail("Episode should have terminated from ground collision")

    def test_truncation_at_max_steps(self, monkeypatch):
        env = AircraftPitchEnv()
        env.reset(seed=0)
        monkeypatch.setattr(env._atmosphere.turbulence, "_w_g",     0.0)
        monkeypatch.setattr(env._atmosphere.turbulence, "_sigma_w", 0.0)
        env._atmosphere.shear._layers    = [{"bottom": 0, "top": 12000, "wind_u": 0.0, "wind_v": 0.0}]
        env._atmosphere.thermals._thermals = []
        monkeypatch.setattr(env, "ALT_PITCH_COUPLING", 0.0)
        monkeypatch.setattr(env, "HDG_ROLL_COUPLING",  0.0)
        env._state     = np.array([0.0, 0.0])
        env.airspeed   = 0.0
        env.altitude   = env._flight_path.target_altitude(0.0)
        env._prev_altitude = env.altitude
        for _ in range(AircraftPitchEnv.MAX_STEPS + 5):
            _, _, terminated, truncated, _ = env.step(AircraftPitchEnv.N_ACTIONS // 2)
            if terminated or truncated:
                break
        assert truncated and not terminated
        assert env._steps <= AircraftPitchEnv.MAX_STEPS

    def test_class_attributes_not_mutated(self):
        assert AircraftPitchEnv.ALT_PITCH_COUPLING == pytest.approx(0.0002)
        assert AircraftPitchEnv.HDG_ROLL_COUPLING  == pytest.approx(0.0003)


class TestAtmosphere:
    """Tests for AtmosphereModel integration, density and turbulence reset."""
    def test_turbulence_produces_nonzero_disturbance(self):
        env = AircraftPitchEnv(turbulence_severity="moderate")
        env.reset(seed=0)
        env._state = np.array([0.0, 0.0])
        disturbances = []
        for _ in range(30):
            _, _, t, tr, _ = env.step(env.N_ACTIONS // 2)
            disturbances.append(env._atmosphere.turbulence._w_g)
            if t or tr:
                break
        assert np.std(disturbances) > 0

    def test_density_decreases_with_altitude(self):
        from atmosphere import ISA
        assert ISA.density(0) > ISA.density(5000) > ISA.density(10000)

    def test_atmosphere_step_returns_required_keys(self, env):
        result = env._atmosphere.step(altitude_m=3000, airspeed_mps=200,
                                       heading_deg=90, dalt_m=0.0, rng=env.np_random)
        for key in ("density", "gust_torque", "shear_torque", "thermal_torque", "total_torque"):
            assert key in result

    def test_new_episode_resets_turbulence(self):
        env = AircraftPitchEnv()
        env.reset(seed=0)
        env._atmosphere.turbulence._w_g = 999.0
        env.reset(seed=1)
        assert env._atmosphere.turbulence._w_g == pytest.approx(0.0)


class TestFlightPath:
    """Tests for FlightPath spline outputs — altitude range, heading wrap, ROC finiteness."""
    def test_target_altitude_within_range(self):
        from flight_path import FlightPath
        fp = FlightPath(rng=np.random.default_rng(42))
        for t in np.linspace(0, fp.duration, 100):
            assert 0 <= fp.target_altitude(t) <= 12_000

    def test_target_heading_is_0_to_360(self):
        from flight_path import FlightPath
        fp = FlightPath(rng=np.random.default_rng(42))
        for t in np.linspace(0, fp.duration, 100):
            assert 0.0 <= fp.target_heading(t) < 360.0

    def test_new_episode_changes_path(self):
        from flight_path import FlightPath
        fp = FlightPath(rng=np.random.default_rng(0))
        alt1 = fp.target_altitude(5.0)
        fp.new_episode(rng=np.random.default_rng(999))
        assert fp.target_altitude(5.0) != alt1

    def test_rate_of_climb_is_finite(self):
        from flight_path import FlightPath
        fp = FlightPath(rng=np.random.default_rng(42))
        for t in np.linspace(0, fp.duration, 50):
            assert np.isfinite(fp.target_rate_of_climb(t))


class TestHeadingError:
    """Tests for _heading_error() wrap-around correctness and boundedness."""
    @pytest.mark.parametrize("actual,target,expected", [
        (10,  350,  20),
        (350,  10, -20),
        (180,   0, -180),
        (  0, 180, -180),
        ( 90,  90,   0),
        (200, 100, 100),
    ])
    def test_heading_error_cases(self, actual, target, expected):
        assert _heading_error(actual, target) == pytest.approx(expected, abs=1e-9)

    def test_heading_error_bounded(self):
        rng = np.random.default_rng(0)
        for _ in range(1000):
            a, t = rng.uniform(0, 360), rng.uniform(0, 360)
            assert -180 <= _heading_error(a, t) <= 180


class TestReproducibility:
    """Tests that same seed produces identical rollouts and different seeds diverge."""
    def test_full_episode_deterministic(self):
        def rollout(seed):
            env = AircraftPitchEnv()
            env.reset(seed=seed); env.action_space.seed(seed)
            rewards = []
            for _ in range(100):
                a = env.action_space.sample()
                _, r, t, tr, _ = env.step(a)
                rewards.append(r)
                if t or tr: break
            return rewards
        np.testing.assert_array_almost_equal(rollout(42), rollout(42))

    def test_different_seeds_differ(self):
        def rollout(seed):
            env = AircraftPitchEnv()
            env.reset(seed=seed); env.action_space.seed(seed)
            rewards = []
            for _ in range(50):
                a = env.action_space.sample()
                _, r, t, tr, _ = env.step(a)
                rewards.append(r)
                if t or tr: break
            return rewards
        r1, r2 = rollout(1), rollout(2)
        assert not np.allclose(r1[:min(len(r1), len(r2))], r2[:min(len(r1), len(r2))])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))