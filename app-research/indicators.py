# file: indicators.py (updated)
import numpy as np
import pandas as pd

def compute_indicators(v, dt, params, scenario_meta):
    """
    Menghitung 38 indikator driving cycle dari v(t) + analisis baterai
    """
    m = params.get("mass", 1500)
    g = 9.81
    Crr = params.get("Crr", 0.015)
    Cd = params.get("Cd", 0.29)
    A = params.get("A", 2.2)
    rho = params.get("rho", 1.225)
    grade = scenario_meta.get("grade", np.zeros_like(v))
    
    # Parameter baterai
    battery_capacity = params.get("battery_capacity", 50)  # kWh
    motor_efficiency = params.get("motor_efficiency", 0.85)
    regen_efficiency = params.get("regen_efficiency", 0.70)
    auxiliary_power = params.get("auxiliary_power", 0.5)  # kW untuk AC, dll

    # Turunan
    a = np.gradient(v, dt)
    jerk = np.gradient(a, dt)
    dist = np.cumsum(v * dt)
    dist_km = dist[-1] / 1000

    # Gaya
    F_rr = m * g * Crr * np.cos(np.arctan(grade))
    F_aero = 0.5 * rho * Cd * A * v ** 2
    F_grade = m * g * np.sin(np.arctan(grade))
    F_trac = m * a + F_rr + F_aero + F_grade
    P_wheel = F_trac * v
    P_pos = np.maximum(P_wheel, 0)
    E_Wh = np.sum(P_pos * dt) / 3600
    ev_Wh_per_km = E_Wh / (dist_km + 1e-9)
    peak_power = np.max(np.abs(P_wheel)) / 1000

    # ANALISIS BATERAI
    # Daya motor (traction dan regenerative braking)
    P_motor_traction = np.where(P_wheel > 0, P_wheel / motor_efficiency, 0)
    P_motor_regen = np.where(P_wheel < 0, P_wheel * regen_efficiency, 0)
    P_motor_total = P_motor_traction + P_motor_regen
    
    # Daya auxiliary (konstan)
    P_auxiliary = auxiliary_power * 1000  # Convert to W
    
    # Daya total dari baterai
    P_battery = np.maximum(P_motor_total + P_auxiliary, 0)
    
    # Energi dari baterai
    E_battery_Wh = np.sum(P_battery * dt) / 3600
    
    # Energi regeneratif
    E_regen_Wh = -np.sum(np.minimum(P_motor_regen, 0) * dt) / 3600
    
    # Konsumsi energi baterai per km
    battery_Wh_per_km = E_battery_Wh / (dist_km + 1e-9)
    
    # State of Charge (SOC) analysis
    initial_soc = params.get("initial_soc", 100)  # %
    battery_capacity_Wh = battery_capacity * 1000
    soc_consumed = (E_battery_Wh / battery_capacity_Wh) * 100
    final_soc = max(0, initial_soc - soc_consumed)
    
    # C-rate analysis
    P_battery_kW = P_battery / 1000
    c_rate_peak = np.max(P_battery_kW) / battery_capacity
    c_rate_avg = np.mean(P_battery_kW) / battery_capacity
    
    # Battery stress indicators
    high_power_events = np.sum(P_battery_kW > battery_capacity * 0.8)  # Events > 0.8C
    battery_stress = np.mean(P_battery_kW > battery_capacity * 0.5) #  % waktu > 0.5C

    # State pengemudi
    is_idle = (v < 0.5).astype(int)
    is_accel = (a > 0.2).astype(int)
    is_decel = (a < -0.2).astype(int)
    is_cruise = ((np.abs(a) <= 0.1) & (v > 0.5)).astype(int)

    def rms(x): return np.sqrt(np.mean(x**2))

    indicators = {
        "avg_speed": np.mean(v),
        "max_speed": np.max(v),
        "min_speed": np.min(v),
        "accel_var": np.var(a),
        "jerk_rms": rms(jerk),
        "distance_km": dist_km,
        "ev_Wh_per_km": ev_Wh_per_km,
        "peak_power_kW": peak_power,
        "idle_ratio": np.mean(is_idle),
        "accel_ratio": np.mean(is_accel),
        "decel_ratio": np.mean(is_decel),
        "cruise_ratio": np.mean(is_cruise),
        "F_rr_mean": np.mean(F_rr),
        "F_aero_mean": np.mean(F_aero),
        "F_grade_mean": np.mean(F_grade),
        "E_total_Wh": E_Wh,
        "energy_efficiency": 1 - (np.mean(F_aero) / np.mean(F_trac + 1e-9)),
        "grade_mean": np.mean(grade),
        "grade_std": np.std(grade),
        "traffic_density": scenario_meta.get("traffic_density", 0.5),
        "road_type": scenario_meta.get("road_type", "urban"),
        "temperature_C": scenario_meta.get("temperature", 25),
        "weather": scenario_meta.get("weather", "clear"),
        "moving_avg_speed": pd.Series(v).rolling(int(10 / dt)).mean().mean(),
        "moving_avg_accel": pd.Series(a).rolling(int(10 / dt)).mean().mean(),
        "speed_std": np.std(v),
        "accel_std": np.std(a),
        "idle_ratio_window": pd.Series(is_idle).rolling(int(10 / dt)).mean().mean(),
        "stop_duration": np.sum(is_idle) * dt,
        "accel_event_duration": np.sum(is_accel) * dt,
        "total_resistance_mean": np.mean(F_rr + F_aero + F_grade),
        "power_mean_kW": np.mean(P_pos) / 1000,
        "power_std_kW": np.std(P_pos) / 1000,
        "battery_stress_proxy": np.mean(np.abs(P_wheel) > 0.8 * peak_power),
        "eco_index_proxy": 1 / (ev_Wh_per_km * (1 + rms(jerk))),
        "aggressiveness_index": np.mean(a > 1.5),
        "comfort_index": 1 / (1 + rms(jerk)),
        
        # INDIKATOR BATERAI BARU
        "battery_energy_Wh": E_battery_Wh,
        "battery_Wh_per_km": battery_Wh_per_km,
        "regen_energy_Wh": E_regen_Wh,
        "regen_efficiency": E_regen_Wh / (E_battery_Wh + 1e-9),
        "final_soc_percent": final_soc,
        "soc_consumed_percent": soc_consumed,
        "c_rate_peak": c_rate_peak,
        "c_rate_avg": c_rate_avg,
        "battery_stress_high_power_events": high_power_events,
        "battery_stress_time_ratio": battery_stress,
        "auxiliary_energy_Wh": (P_auxiliary * len(v) * dt) / 3600,
        "motor_efficiency_actual": E_Wh / (E_battery_Wh + 1e-9)
    }
    return indicators