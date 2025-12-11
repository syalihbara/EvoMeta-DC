# eco_driving_multidomain_gui_fixed.py

import sys
import numpy as np
from typing import Tuple

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit
)

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

# ============================================================
# 1. DEFINISI LINGKUNGAN MULTI‑DOMAIN (5 DOMAIN)
# ============================================================

N_SEGMENTS = 80
SEGMENT_LENGTH = 100.0  # m

DOMAIN_CONGESTED = 0
DOMAIN_URBAN     = 1
DOMAIN_SUBURBAN  = 2
DOMAIN_HIGHWAY   = 3
DOMAIN_HILLY     = 4

domain_id = np.zeros(N_SEGMENTS, dtype=int)
domain_id[0:15]   = DOMAIN_CONGESTED
domain_id[15:30]  = DOMAIN_URBAN
domain_id[30:45]  = DOMAIN_SUBURBAN
domain_id[45:60]  = DOMAIN_HIGHWAY
domain_id[60:80]  = DOMAIN_HILLY

V_MIN = 0.0
V_MAX_DOMAIN = {
    DOMAIN_CONGESTED: 11.11,  # 40 km/h
    DOMAIN_URBAN:     16.67,  # 60 km/h
    DOMAIN_SUBURBAN:  22.22,  # 80 km/h
    DOMAIN_HIGHWAY:   30.56,  # 110 km/h
    DOMAIN_HILLY:     25.00   # 90 km/h
}

SLOPE_DOMAIN = {
    DOMAIN_CONGESTED: 0.0,
    DOMAIN_URBAN:     0.005,
    DOMAIN_SUBURBAN:  0.01,
    DOMAIN_HIGHWAY:   0.005,
    DOMAIN_HILLY:     0.04
}

# Parameter kendaraan/baterai
VEHICLE_MASS = 1500.0     # kg
AIR_DENSITY = 1.2         # kg/m3
FRONTAL_AREA = 2.2        # m2
DRAG_COEFF = 0.29
ROLLING_RES = 0.015
GRAVITY = 9.81
DRIVETRAIN_EFF = 0.9
BATTERY_CAPACITY = 60_000.0  # Wh
NOMINAL_VOLTAGE = 350.0      # V

# ============================================================
# 2. MODEL ENERGI & DEGRADASI
# ============================================================

def compute_segment_energy(v_prev: float, v_next: float, slope: float = 0.0) -> float:
    v_avg = max(1.0, (v_prev + v_next) / 2.0)  # lindungi dari v_avg sangat kecil
    dt = SEGMENT_LENGTH / v_avg

    F_aero = 0.5 * AIR_DENSITY * FRONTAL_AREA * DRAG_COEFF * v_avg**2
    F_roll = VEHICLE_MASS * GRAVITY * ROLLING_RES * np.cos(slope)
    F_grade = VEHICLE_MASS * GRAVITY * np.sin(slope)

    a = (v_next - v_prev) / dt
    F_inertial = VEHICLE_MASS * a

    F_total = F_aero + F_roll + F_grade + F_inertial
    P_trac = F_total * v_avg

    if P_trac >= 0:
        P_batt = P_trac / DRIVETRAIN_EFF
    else:
        P_batt = 0.3 * P_trac * DRIVETRAIN_EFF

    E_Wh = P_batt * dt / 3600.0
    return E_Wh


def compute_battery_degradation_metric(power_profile: np.ndarray) -> float:
    I = power_profile / max(1.0, NOMINAL_VOLTAGE)
    dIdt = np.diff(I)
    stress = np.sum(np.abs(dIdt))
    return float(np.clip(stress, -1e9, 1e9))

# ============================================================
# 3. SIMULASI MULTI‑DOMAIN
# ============================================================

def simulate_trajectory_multidomain(v_profile: np.ndarray) -> Tuple[float, float, float]:
    energies = []
    power_profile = []
    E_cong_urban = 0.0

    for i in range(N_SEGMENTS):
        dom = domain_id[i]
        slope = SLOPE_DOMAIN[dom]

        v_prev = v_profile[i]
        v_next = v_profile[i+1]
        E = compute_segment_energy(v_prev, v_next, slope)
        energies.append(E)

        v_avg = max(1.0, (v_prev + v_next) / 2.0)
        dt = SEGMENT_LENGTH / v_avg
        P = E * 3600.0 / max(1e-3, dt)
        power_profile.append(P)

        if dom in (DOMAIN_CONGESTED, DOMAIN_URBAN):
            E_cong_urban += E

    E_total = np.sum(energies)
    degr = compute_battery_degradation_metric(np.array(power_profile))

    # clamp agar tidak ekstrem
    E_total = float(np.clip(E_total, -1e9, 1e9))
    E_cong_urban = float(np.clip(E_cong_urban, -1e9, 1e9))

    return E_total, E_cong_urban, degr

# ============================================================
# 4. MASALAH MANY‑OBJECTIVE UNTUK PY MOO
# ============================================================

class EcoDrivingMultiDomainProblem(Problem):
    def __init__(self):
        n_var = N_SEGMENTS + 1
        n_obj = 3
        n_constr = 0

        xl = np.zeros(n_var)
        xu = np.zeros(n_var)
        for i in range(n_var):
            dom = domain_id[min(i, N_SEGMENTS-1)]
            vmax = V_MAX_DOMAIN[dom]
            xl[i] = V_MIN
            xu[i] = vmax

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for row in X:
            v_profile = row.copy()
            v_profile[0] = 0.0
            v_profile[-1] = 0.0
            E_tot, E_cu, degr = simulate_trajectory_multidomain(v_profile)
            F.append([E_tot, E_cu, degr])
        out["F"] = np.array(F)

# ============================================================
# 5. HONEY BADGER LOCAL SEARCH (HBOA)
# ============================================================

def hboa_local_search(X: np.ndarray, F: np.ndarray,
                      n_iters: int = 5,
                      alpha: float = 0.8,
                      beta: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    pop_size = X.shape[0]
    n_elite = max(2, pop_size // 5)

    idx_sort = np.argsort(F[:, 0])
    elite_idx = idx_sort[:n_elite]

    for idx in elite_idx:
        x_best = X[idx].copy()
        f_best = F[idx].copy()

        for _ in range(n_iters):
            noise = np.random.normal(0, 1.0, size=x_best.shape)
            cand = x_best + alpha * noise + beta * np.random.uniform(-1, 1, size=x_best.shape)

            for i in range(len(cand)):
                dom = domain_id[min(i, N_SEGMENTS-1)]
                vmax = V_MAX_DOMAIN[dom]
                cand[i] = np.clip(cand[i], V_MIN, vmax)

            cand[0] = 0.0
            cand[-1] = 0.0

            E_tot, E_cu, degr = simulate_trajectory_multidomain(cand)
            f_cand = np.array([E_tot, E_cu, degr])

            if np.all(f_cand <= f_best) and np.any(f_cand < f_best):
                x_best = cand
                f_best = f_cand

        X[idx] = x_best
        F[idx] = f_best

    return X, F

# ============================================================
# 6. FUNGSI OPTIMASI HYBRID NSGA‑III + HBOA
# ============================================================

def run_hybrid_nsga3_hboa(pop_size: int, n_gen: int):
    problem = EcoDrivingMultiDomainProblem()

    # reference directions yang lebih sedikit (n_partitions=3 → 10 ref_dirs untuk 3 objektif)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=3)
    # pastikan pop_size >= jumlah ref_dirs
    if pop_size < len(ref_dirs):
        pop_size = len(ref_dirs)

    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    termination = get_termination("n_gen", n_gen)

    res = minimize(problem,
                   algorithm,
                   termination=termination,
                   save_history=False,
                   verbose=False)

    X = res.pop.get("X")
    F = res.pop.get("F")

    # refine dengan HBOA
    X_refined, F_refined = hboa_local_search(X, F, n_iters=5)
    return X_refined, F_refined

# ============================================================
# 7. GUI PyQt5
# ============================================================

class EcoDrivingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Hybrid NSGA-III + HBOA Eco-Driving Multi-Domain EV")

        layout = QVBoxLayout()

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Pop Size:"))
        self.pop_input = QLineEdit("40")
        h1.addWidget(self.pop_input)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Generations:"))
        self.gen_input = QLineEdit("30")
        h2.addWidget(self.gen_input)

        layout.addLayout(h1)
        layout.addLayout(h2)

        self.run_button = QPushButton("Run Optimization")
        self.run_button.clicked.connect(self.run_optimization)
        layout.addWidget(self.run_button)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        self.setLayout(layout)

    def run_optimization(self):
        try:
            pop_size = int(self.pop_input.text())
            n_gen = int(self.gen_input.text())
        except ValueError:
            self.output_text.setText("Pop Size dan Generations harus berupa integer.")
            return

        self.output_text.setText("Menjalankan optimasi...\nMohon tunggu...\n")

        X, F = run_hybrid_nsga3_hboa(pop_size, n_gen)

        txt = "Hasil optimasi (beberapa solusi Pareto):\n"
        idx_sort = np.argsort(F[:, 0])
        top_k = min(5, len(idx_sort))
        for rank in range(top_k):
            i = idx_sort[rank]
            E_tot, E_cu, degr = F[i]
            txt += (f"Solusi {rank+1}:\n"
                    f"  Energi total        = {E_tot:.2f} Wh\n"
                    f"  Energi cong+urban   = {E_cu:.2f} Wh\n"
                    f"  Degr. baterai (proxy)= {degr:.2f}\n")

        self.output_text.setText(txt)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = EcoDrivingGUI()
    gui.resize(600, 400)
    gui.show()
    sys.exit(app.exec_())
