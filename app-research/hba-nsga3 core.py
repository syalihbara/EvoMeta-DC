import numpy as np
import threading
from objective_functions import N_INDICATORS, N_OBJECTIVES, LB, UB, evaluate_solution
from hba_operator import calculate_hba_params, hba_mutation

def initialize_population(pop_size):
    """Inisialisasi populasi acak (38D)."""
    pop = []
    for _ in range(pop_size):
        position = LB + np.random.rand(N_INDICATORS) * (UB - LB)
        fitness = evaluate_solution(position)
        pop.append({'position': position, 'fitness': fitness})
    return pop

def nondominated_sort_simplified(population):
    """
    PROXI NSGA-III: Mengembalikan solusi terbaik berdasarkan fitness sum terkecil (Minimasi).
    Dalam implementasi NSGA-III asli, ini adalah proses Non-Dominated Sorting & Niching kompleks.
    """
    if not population:
        return None
    fitness_sums = [np.sum(ind['fitness']) for ind in population]
    best_idx = np.argmin(fitness_sums)
    return population[best_idx]

def run_optimization(pop_size, t_max, update_gui_log, stop_event):
    """Menjalankan algoritma NSGA-III-HBA."""
    
    # 1. INISIALISASI
    population = initialize_population(pop_size)
    best_solution = nondominated_sort_simplified(population)

    update_gui_log(f"Optimization Started (D={N_INDICATORS}, M={N_OBJECTIVES})")
    
    for t in range(1, t_max + 1):
        if stop_event.is_set():
            update_gui_log("Optimasi Dihentikan oleh Pengguna.")
            break
            
        # 2. GENERASI KETURUNAN (Q_t) via HBA MUTATION
        offspring_pop = []
        parent_pop = population 
        
        I_values, alpha = calculate_hba_params([ind['position'] for ind in parent_pop], 
                                                best_solution, t, t_max)
        
        for i in range(pop_size):
            x_i = parent_pop[i]['position']
            I_i = I_values[i]
            
            # Mutasi HBA (38D)
            new_position = hba_mutation(x_i, best_solution['position'], I_i, alpha)
            new_fitness = evaluate_solution(new_position)
            
            offspring_pop.append({'position': new_position, 'fitness': new_fitness})

        # 3. GABUNGKAN DAN SELEKSI NSGA-III (DISEDERHANAKAN)
        R_t = population + offspring_pop
        best_solution_R = nondominated_sort_simplified(R_t)
        
        if np.sum(best_solution_R['fitness']) < np.sum(best_solution['fitness']):
             best_solution = best_solution_R
        
        population = R_t[:pop_size] # Ambil kembali populasi berukuran N

        log_msg = f"Iterasi {t}/{t_max} | Best Sum Fitness: {np.sum(best_solution['fitness']):.4f} | Alpha: {alpha:.4f}"
        update_gui_log(log_msg)

    # 4. HASIL AKHIR
    final_result = {
        'status': 'Finished',
        'best_position': best_solution['position'],
        'best_fitness': best_solution['fitness']
    }
    
    if not stop_event.is_set():
        update_gui_log("\n--- Optimasi Selesai ---")
    update_gui_log(f"Best Solution (38D): {final_result['best_position'][:5]}... (5 dari 38 Indikator)")
    update_gui_log(f"Objective Values (M=3): {final_result['best_fitness']}")
    
    return final_result
