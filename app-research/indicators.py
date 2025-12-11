"""
indicators.py
Compute indicators for driving cycles, plus EDI and BMSI utilities.
"""
import numpy as np
import pandas as pd

def compute_indicators(v, dt=1.0, params=None, scenario_meta=None):
    if params is None: params = {}
    if scenario_meta is None: scenario_meta = {}
    m = params.get("mass", 1300.0)
    g = 9.81
    Crr = params.get("Crr", 0.015)
    Cd = params.get("Cd", 0.29)
    A = params.get("A", 2.2)
    rho = params.get("rho", 1.225)
    eta_motor = params.get("eta_motor", 0.90)

    grade = np.array(scenario_meta.get("grade", np.zeros_like(v)))
    time = np.arange(0, len(v)*dt, dt)

    a = np.gradient(v, dt)
    jerk = np.gradient(a, dt)
    distance = np.cumsum(v * dt)
    distance_km = distance[-1] / 1000.0 if distance[-1] > 0 else 0.0

    F_rr = m * g * Crr * np.cos(np.arctan(grade))
    F_aero = 0.5 * rho * Cd * A * (v**2)
    F_grade = m * g * np.sin(np.arctan(grade))
    F_trac = m * a + F_rr + F_aero + F_grade

    P_wheel = F_trac * v
    P_wheel_pos = np.maximum(P_wheel, 0.0)
    E_input_Wh = np.sum(P_wheel_pos * dt) / 3600.0
    ev_Wh_per_km = (E_input_Wh / distance_km) if distance_km > 0 else np.nan
    peak_power_kW = np.max(np.abs(P_wheel))/1000.0 if len(P_wheel)>0 else 0.0

    is_idle = (v < 0.5).astype(float)
    is_accel = (a > 0.2).astype(float)
    is_decel = (a < -0.2).astype(float)
    is_cruise = ((np.abs(a) <= 0.2) & (v > 0.5)).astype(float)

    def rms(x): return float(np.sqrt(np.mean(np.array(x)**2)))
    def pct(x): return float(np.mean(np.array(x)))

    win_s = int(max(1, round(10.0/dt)))
    ser_v = pd.Series(v)
    ser_a = pd.Series(a)
    moving_avg_speed = ser_v.rolling(win_s, min_periods=1).mean().values
    moving_avg_accel = ser_a.rolling(win_s, min_periods=1).mean().values

    indicators = {
        "time_s_total": float(time[-1]) if len(time)>0 else 0.0,
        "dt_s": float(dt),
        "distance_m": float(distance[-1]),
        "elapsed_time_s": float(time[-1]) if len(time)>0 else 0.0,
        "speed_mps_mean": float(np.mean(v)),
        "speed_kph_mean": float(np.mean(v)*3.6),
        "accel_mps2_mean": float(np.mean(a)),
        "jerk_mps3_rms": float(rms(jerk)),
        "max_speed_mps": float(np.max(v)) if len(v)>0 else 0.0,
        "avg_speed_window_mps": float(np.mean(moving_avg_speed)),
        "accel_variance": float(np.var(a)),
        "is_idle_ratio": float(pct(is_idle)),
        "is_accel_ratio": float(pct(is_accel)),
        "is_decel_ratio": float(pct(is_decel)),
        "is_cruise_ratio": float(pct(is_cruise)),
        "stop_duration_s": float(np.sum(is_idle) * dt),
        "accel_event_duration_s": float(np.sum(is_accel) * dt),
        "tractive_effort_N_mean": float(np.mean(F_trac)),
        "wheel_power_proxy_kW_mean": float(np.mean(P_wheel)/1000.0),
        "wheel_power_proxy_kW_pos_mean": float(np.mean(P_wheel_pos)/1000.0),
        "energy_input_Wh_total": float(E_input_Wh),
        "rolling_resistance_N_mean": float(np.mean(F_rr)),
        "aero_drag_N_mean": float(np.mean(F_aero)),
        "total_resistance_N_mean": float(np.mean(F_rr + F_aero + F_grade)),
        "road_grade_mean": float(np.mean(grade)),
        "curvature_mean": float(np.mean(scenario_meta.get('curvature', np.zeros_like(v)))),
        "road_type": str(scenario_meta.get('road_type', 'generic')),
        "traffic_density_index": float(scenario_meta.get('traffic_density', 0.5)),
        "weather_condition": str(scenario_meta.get('weather', 'clear')),
        "temperature_C": float(scenario_meta.get('temperature', 25.0)),
        "moving_avg_speed_mean": float(np.mean(moving_avg_speed)),
        "moving_avg_accel_mean": float(np.mean(moving_avg_accel)),
        "speed_std": float(np.std(v)),
        "accel_std": float(np.std(a)),
        "idle_ratio_window": float(np.mean(pd.Series(is_idle).rolling(win_s, min_periods=1).mean().values)),
        "jerk_rms": float(rms(jerk)),
        "aggressiveness_index": float(np.mean(np.abs(a) > 2.0)),
        "ev_Wh_per_km": float(ev_Wh_per_km) if not np.isnan(ev_Wh_per_km) else np.nan,
        "peak_power_kW": float(peak_power_kW),
        "power_mean_kW": float(np.mean(P_wheel_pos)/1000.0),
        "battery_stress_time_s": float(np.sum((np.abs(P_wheel) > 0.6 * peak_power_kW * 1000.0).astype(float)) * dt),
        "energy_efficiency_proxy": float(1.0 - (np.mean(F_aero) / (np.mean(np.abs(F_trac)) + 1e-9)))
    }
    return indicators

def calc_EDI(df, weights=None):
    if weights is None:
        weights = {"energy":0.4, "jerk":0.3, "accel":0.2, "idle":0.1}
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    df = df.copy()
    E = 1.0 / (df['ev_Wh_per_km'].astype(float) + 1e-9)
    J = 1.0 / (df['jerk_rms'].astype(float) + 1e-9)
    A = 1.0 / (np.abs(df['accel_mps2_mean'].astype(float)) + 1e-9)
    S = 1.0 - df['is_idle_ratio'].astype(float)
    def norm(s):
        s = s.astype(float)
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)
    En = norm(E); Jn = norm(J); An = norm(A); Sn = norm(S)
    df['EDI'] = weights['energy']*En + weights['jerk']*Jn + weights['accel']*An + weights['idle']*Sn
    return df

def calc_BMSI(cycle_df, params=None):
    if params is None:
        params = {}
    soc = cycle_df.get('soc', None)
    if soc is None:
        soc = np.linspace(1.0, np.random.uniform(0.2, 0.95), len(cycle_df))
    I = cycle_df.get('current_A', None)
    if I is None:
        I = np.random.normal(50, 10, len(soc))
    T = cycle_df.get('temp_C', np.ones_like(I)*25.0)
    delta_soc = float(np.nanmax(soc) - np.nanmin(soc))
    sigma_I = float(np.std(I))
    Tavg = float(np.mean(T))
    DOD = delta_soc * 100.0
    delta_soc_max = params.get('delta_soc_max', 0.8)
    sigma_I_max = params.get('sigma_I_max', 200.0)
    T_max = params.get('T_max', 60.0)
    DOD_max = params.get('DOD_max', 80.0)
    score = (
        0.25 * max(0.0, 1.0 - (delta_soc / delta_soc_max)) +
        0.25 * max(0.0, 1.0 - (sigma_I / sigma_I_max)) +
        0.25 * max(0.0, 1.0 - (Tavg / T_max)) +
        0.25 * max(0.0, 1.0 - (DOD / DOD_max))
    )
    eta = 1.0 - min(0.9, sigma_I / (abs(np.mean(I)) + 1e-9)) * 0.1
    bmsi = float(max(0.0, min(1.0, score * eta)))
    return bmsi
