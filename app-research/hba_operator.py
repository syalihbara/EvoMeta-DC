import numpy as np
from objective_functions import N_INDICATORS, LB, UB

# Konstanta HBA
BETA = 6      
C = 2         

def calculate_hba_params(current_pop, best_solution, t, t_max):
    """Menghitung Intensitas Aroma (I) dan Faktor Kepadatan (alpha)."""
    
    alpha = C * np.exp(-t / t_max)
    I_values = []
    x_prey = best_solution['position']
    
    for x_i in current_pop:
        # Perhitungan Jarak Euklides di ruang 38D
        d_i = np.linalg.norm(x_prey - x_i) 
        
        if d_i < 1e-10:
             I_i = 1.0 
        else:
             r2 = np.random.rand()
             S = 1.0 # Kekuatan Sumber S diasumsikan 1
             I_i = (r2**2) * (S / (4 * np.pi * d_i**2))
             
        I_values.append(I_i)
        
    return np.array(I_values), alpha

def hba_mutation(x_i, x_prey, I_i, alpha):
    """Menerapkan pembaruan posisi HBA (Digging/Honey Phase) (38D)."""
    
    r_val = np.random.rand() 
    r3 = np.random.rand()    
    r4 = np.random.rand()    
    
    # Vektor Arah Bendera F (38 dimensi)
    F_vec = (-1.0) ** (np.random.randint(0, 2, size=N_INDICATORS) * np.round(r4 * 2))
    
    # Vektor Digging/Honey (Vektor 38D)
    D_H_vec = x_prey - x_i
    
    if r_val < 0.5:
        # Digging Phase (Eksplorasi)
        x_new = x_prey + F_vec * BETA * I_i * x_prey + F_vec * r3 * alpha * D_H_vec
    else:
        # Honey Phase (Eksploitasi)
        x_new = x_prey + F_vec * r3 * alpha * D_H_vec
        
    # Boundary Check untuk 38 dimensi
    x_new = np.clip(x_new, LB, UB)
    
    return x_new