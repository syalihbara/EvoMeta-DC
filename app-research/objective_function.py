import numpy as np

# Dimensi masalah
N_INDICATORS = 38
N_OBJECTIVES = 3

# Batasan untuk setiap indikator (asumsi batas seragam [0, 1] untuk kesederhanaan)
LB = np.zeros(N_INDICATORS)
UB = np.ones(N_INDICATORS)

def evaluate_solution(solution_vector):
    """
    Evaluasi solusi 38D terhadap 3 fungsi objektif.
    
    Arg:
        solution_vector (np.array): Vektor posisi individu (38 indikator).
        
    Ret:
        np.array: Vektor nilai objektif (f1, f2, f3).
    """
    # -----------------------------------------------------------
    # CONTOH: Fungsi DTLZ2 yang diubah (populer di Many-Objective Optimization)
    # -----------------------------------------------------------
    
    # 1. Hitung g(x) (variabel yang tidak mempengaruhi fungsi objektif pertama M-1)
    # Asumsi: g hanya bergantung pada 3 indikator terakhir
    g = 0.0
    for x_j in solution_vector[N_INDICATORS-3:]:
        g += (x_j - 0.5)**2
    
    # 2. Hitung Fungsi Objektif (Minimasi)
    
    # f1: f1 = (1 + g) * cos(x1) * cos(x2) * ...
    # Kita menggunakan f1 = (1 + g) * perkalian dari (M-1) indikator pertama
    f1 = (1.0 + g) * np.prod(np.cos(solution_vector[:N_OBJECTIVES-1] * np.pi / 2.0))
    
    # f2: f2 = (1 + g) * cos(x1) * cos(x2) * ... * sin(x(M-1))
    f2 = (1.0 + g) * np.prod(np.cos(solution_vector[:N_OBJECTIVES-2] * np.pi / 2.0)) * \
         np.sin(solution_vector[N_OBJECTIVES-2] * np.pi / 2.0)
         
    # f3: f3 = (1 + g) * sin(x1)
    f3 = (1.0 + g) * np.sin(solution_vector[0] * np.pi / 2.0)
    
    return np.array([f1, f2, f3])