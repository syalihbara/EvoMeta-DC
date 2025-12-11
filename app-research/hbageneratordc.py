import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Callable, Dict
import time
import warnings
warnings.filterwarnings('ignore')

class HoneyBadgerAlgorithm:
    """
    Honey Badger Algorithm (HBA) untuk optimasi
    Paper asli: "Honey Badger Algorithm: A New Metaheuristic Algorithm for Solving Optimization Problems"
    """
    
    def __init__(self, objective_func: Callable, dim: int, bounds: List[Tuple[float, float]], 
                 population_size: int = 50, max_iter: int = 100, seed: int = None):
        """
        Inisialisasi HBA
        
        Parameters:
        - objective_func: Fungsi tujuan yang akan dioptimasi
        - dim: Dimensi masalah
        - bounds: Batas untuk setiap dimensi [(min, max), ...]
        - population_size: Ukuran populasi
        - max_iter: Maksimum iterasi
        - seed: Random seed untuk reproduktibilitas
        """
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Parameter HBA
        self.beta = 6.0      # Kemampuan eksplorasi
        self.C = 2.0         # Konstanta intensitas
        self.epsilon = 1e-10 # Untuk menghindari division by zero
        
        # Inisialisasi populasi
        self.positions = self.initialize_population()
        self.fitness = np.array([self.objective_func(pos) for pos in self.positions])
        
        # Best solution
        self.best_position = self.positions[np.argmin(self.fitness)].copy()
        self.best_fitness = np.min(self.fitness)
        
        # History untuk analisis
        self.convergence_curve = []
        self.position_history = []
        self.fitness_history = []
    
    def initialize_population(self) -> np.ndarray:
        """Inisialisasi populasi awal secara acak dalam batas"""
        positions = np.zeros((self.pop_size, self.dim))
        for i in range(self.dim):
            positions[:, i] = np.random.uniform(
                self.bounds[i, 0], 
                self.bounds[i, 1], 
                self.pop_size
            )
        return positions
    
    def calculate_intensity(self, fitness: np.ndarray) -> np.ndarray:
        """Menghitung intensitas berdasarkan fitness"""
        # Intensitas berbanding terbalik dengan fitness
        # Fitness lebih kecil = intensitas lebih tinggi
        fitness = np.array(fitness)
        
        # Hindari division by zero
        fitness = np.where(fitness == 0, self.epsilon, fitness)
        
        # Normalisasi fitness
        min_fit = np.min(fitness)
        max_fit = np.max(fitness)
        
        if max_fit == min_fit:
            return np.ones_like(fitness)
        
        intensity = (max_fit - fitness + self.epsilon) / (max_fit - min_fit + self.epsilon)
        return intensity
    
    def update_density_factor(self, iter: int) -> float:
        """Update density factor (menurun seiring waktu)"""
        return self.C * np.exp(-iter / self.max_iter)
    
    def update_positions(self, iter: int):
        """Update posisi honey badger"""
        # Hitung intensitas
        I = self.calculate_intensity(self.fitness)
        density = self.update_density_factor(iter)
        
        new_positions = np.zeros_like(self.positions)
        
        for i in range(self.pop_size):
            # Fase 1: Digging phase (eksplorasi)
            F = np.random.choice([-1, 1])  # Direction factor
            
            # Update berdasarkan digging behavior
            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
            r4 = np.random.random()
            
            # Digging component
            digging_component = F * self.beta * I[i] * self.best_position
            digging_component += F * r1 * density * self.positions[i]
            digging_component -= r2 * self.best_position
            
            # Fase 2: Honey phase (eksploitasi)
            honey_component = r3 * density * (self.positions[i] - self.best_position)
            
            # Combine phases dengan probabilitas
            if r4 < 0.5:
                # Lebih fokus pada digging
                new_pos = self.best_position + digging_component + honey_component
            else:
                # Lebih fokus pada honey
                new_pos = self.best_position + honey_component
            
            # Apply bounds
            for d in range(self.dim):
                new_pos[d] = np.clip(new_pos[d], self.bounds[d, 0], self.bounds[d, 1])
            
            new_positions[i] = new_pos
        
        # Evaluasi fitness baru
        new_fitness = np.array([self.objective_func(pos) for pos in new_positions])
        
        # Update jika fitness lebih baik
        for i in range(self.pop_size):
            if new_fitness[i] < self.fitness[i]:
                self.positions[i] = new_positions[i]
                self.fitness[i] = new_fitness[i]
        
        # Update best solution
        current_best_idx = np.argmin(self.fitness)
        current_best_fitness = self.fitness[current_best_idx]
        
        if current_best_fitness < self.best_fitness:
            self.best_position = self.positions[current_best_idx].copy()
            self.best_fitness = current_best_fitness
        
        # Simpan history
        self.convergence_curve.append(self.best_fitness)
        self.position_history.append(self.positions.copy())
        self.fitness_history.append(self.fitness.copy())
    
    def optimize(self) -> Dict:
        """Menjalankan optimasi HBA"""
        print("🚀 Memulai Honey Badger Algorithm...")
        start_time = time.time()
        
        for iter in range(self.max_iter):
            self.update_positions(iter)
            
            if (iter + 1) % 10 == 0:
                print(f"Iterasi {iter + 1}/{self.max_iter}, Best Fitness: {self.best_fitness:.6f}")
        
        end_time = time.time()
        
        result = {
            'best_position': self.best_position,
            'best_fitness': self.best_fitness,
            'convergence_curve': self.convergence_curve,
            'position_history': self.position_history,
            'fitness_history': self.fitness_history,
            'execution_time': end_time - start_time,
            'iterations': self.max_iter
        }
        
        print(f"✅ Optimasi selesai dalam {end_time - start_time:.2f} detik")
        print(f"🎯 Best Fitness: {self.best_fitness:.6f}")
        print(f"📍 Best Position: {self.best_position}")
        
        return result

# ==================== FUNGSI TEST BENCHMARK ====================

class BenchmarkFunctions:
    """Kumpulan fungsi benchmark untuk testing HBA"""
    
    @staticmethod
    def sphere(x):
        """Sphere Function - Minimum di [0,0,...,0]"""
        return np.sum(x**2)
    
    @staticmethod
    def rastrigin(x):
        """Rastrigin Function - Minimum di [0,0,...,0]"""
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def rosenbrock(x):
        """Rosenbrock Function - Minimum di [1,1,...,1]"""
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    @staticmethod
    def ackley(x):
        """Ackley Function - Minimum di [0,0,...,0]"""
        a = 20
        b = 0.2
        c = 2 * np.pi
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.exp(1)
    
    @staticmethod
    def griewank(x):
        """Griewank Function - Minimum di [0,0,...,0]"""
        sum_part = np.sum(x**2) / 4000
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_part - prod_part + 1
    
    @staticmethod
    def schwefel(x):
        """Schwefel Function - Minimum di [420.9687,...,420.9687]"""
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    @staticmethod
    def zakharov(x):
        """Zakharov Function - Minimum di [0,0,...,0]"""
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * np.arange(1, n + 1) * x)
        return sum1 + sum2**2 + sum2**4

# ==================== VISUALIZATION ====================

class HBAAnalyzer:
    """Class untuk menganalisis dan memvisualisasikan hasil HBA"""
    
    @staticmethod
    def plot_convergence(result: Dict, title: str = "HBA Convergence"):
        """Plot kurva konvergensi"""
        plt.figure(figsize=(10, 6))
        plt.plot(result['convergence_curve'], 'b-', linewidth=2)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_search_pattern(result: Dict, bounds: List[Tuple[float, float]], 
                          function_name: str = "Function"):
        """Plot pola pencarian untuk fungsi 2D"""
        if len(bounds) != 2:
            print("Plot search pattern hanya untuk fungsi 2D")
            return
        
        # Generate function landscape
        x = np.linspace(bounds[0][0], bounds[0][1], 100)
        y = np.linspace(bounds[1][0], bounds[1][1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = BenchmarkFunctions.sphere([X[i, j], Y[i, j]])
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Plot contour
        contour = plt.contour(X, Y, Z, levels=50, alpha=0.6)
        plt.clabel(contour, inline=True, fontsize=8)
        
        # Plot search positions
        position_history = result['position_history']
        colors = plt.cm.viridis(np.linspace(0, 1, len(position_history)))
        
        for iter_idx, positions in enumerate(position_history[::5]):  # Sample every 5 iterations
            plt.scatter(positions[:, 0], positions[:, 1], 
                       color=colors[iter_idx], alpha=0.6, s=20,
                       label=f'Iter {iter_idx*5}' if iter_idx % 5 == 0 else "")
        
        # Plot best solution
        best_pos = result['best_position']
        plt.scatter(best_pos[0], best_pos[1], color='red', s=200, 
                   marker='*', edgecolors='black', linewidth=2, 
                   label='Best Solution')
        
        plt.title(f'HBA Search Pattern - {function_name}', fontsize=14, fontweight='bold')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.colorbar(contour, label='Fitness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compare_algorithms(results_dict: Dict, title: str = "Algorithm Comparison"):
        """Membandingkan beberapa algoritma"""
        plt.figure(figsize=(12, 8))
        
        for algo_name, result in results_dict.items():
            plt.plot(result['convergence_curve'], label=algo_name, linewidth=2)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def generate_report(results: Dict, function_name: str) -> pd.DataFrame:
        """Generate laporan performa"""
        report_data = {
            'Function': [function_name],
            'Best Fitness': [results['best_fitness']],
            'Execution Time (s)': [results['execution_time']],
            'Iterations': [results['iterations']],
            'Best Solution': [str(results['best_position'])]
        }
        
        return pd.DataFrame(report_data)

# ==================== APLIKASI: OPTIMASI DRIVING CYCLE ====================

class DrivingCycleOptimizer:
    """Mengaplikasikan HBA untuk optimasi driving cycle"""
    
    def __init__(self, target_cycle: np.ndarray, cycle_duration: int = 1800):
        self.target_cycle = target_cycle
        self.cycle_duration = cycle_duration
        self.dim = cycle_duration  # Setiap time step adalah sebuah dimensi
    
    def fitness_function(self, generated_cycle: np.ndarray) -> float:
        """Fungsi fitness: meminimalkan perbedaan dengan target cycle"""
        
        # Pastikan panjang cycle sama
        if len(generated_cycle) != len(self.target_cycle):
            generated_cycle = np.interp(
                np.linspace(0, len(generated_cycle)-1, len(self.target_cycle)),
                np.arange(len(generated_cycle)),
                generated_cycle
            )
        
        # Hitung Mean Squared Error
        mse = np.mean((generated_cycle - self.target_cycle)**2)
        
        # Penalty untuk perubahan akselerasi yang tidak realistis
        acceleration = np.diff(generated_cycle)
        max_accel_penalty = np.sum(np.maximum(np.abs(acceleration) - 3.0, 0)**2)
        
        # Penalty untuk speed di luar batas realistis
        speed_penalty = np.sum(np.maximum(generated_cycle - 130, 0)**2) + \
                       np.sum(np.maximum(-generated_cycle, 0)**2)
        
        # Total fitness
        total_fitness = mse + 0.1 * max_accel_penalty + 0.1 * speed_penalty
        
        return total_fitness
    
    def optimize_cycle(self, population_size: int = 30, max_iter: int = 50):
        """Optimasi driving cycle menggunakan HBA"""
        
        # Batas speed: 0-130 km/h
        bounds = [(0, 130) for _ in range(self.dim)]
        
        # Inisialisasi HBA
        hba = HoneyBadgerAlgorithm(
            objective_func=self.fitness_function,
            dim=self.dim,
            bounds=bounds,
            population_size=population_size,
            max_iter=max_iter,
            seed=42
        )
        
        # Jalankan optimasi
        result = hba.optimize()
        
        return result, hba

# ==================== CONTOH PENGGUNAAN ====================

def demo_benchmark_functions():
    """Demo HBA pada berbagai fungsi benchmark"""
    
    # Konfigurasi
    dim = 10
    pop_size = 50
    max_iter = 100
    bounds = [(-5.12, 5.12) for _ in range(dim)]  # Batas untuk kebanyakan fungsi
    
    # Daftar fungsi benchmark
    benchmark_functions = {
        'Sphere': BenchmarkFunctions.sphere,
        'Rastrigin': BenchmarkFunctions.rastrigin,
        'Rosenbrock': BenchmarkFunctions.rosenbrock,
        'Ackley': BenchmarkFunctions.ackley
    }
    
    results = {}
    
    for func_name, func in benchmark_functions.items():
        print(f"\n{'='*50}")
        print(f"🔬 Testing HBA on {func_name} Function")
        print(f"{'='*50}")
        
        # Jalankan HBA
        hba = HoneyBadgerAlgorithm(
            objective_func=func,
            dim=dim,
            bounds=bounds,
            population_size=pop_size,
            max_iter=max_iter,
            seed=42
        )
        
        result = hba.optimize()
        results[func_name] = result
        
        # Plot convergence
        HBAAnalyzer.plot_convergence(result, f"HBA on {func_name} Function")
    
    return results

def demo_driving_cycle_optimization():
    """Demo optimasi driving cycle"""
    
    print(f"\n{'='*60}")
    print("🚗 DEMO: Driving Cycle Optimization dengan HBA")
    print(f"{'='*60}")
    
    # Buat target cycle sintetis (contoh: urban driving cycle)
    duration = 300  # 5 menit untuk demo (300 detik)
    t = np.linspace(0, 4*np.pi, duration)
    
    # Target cycle: kombinasi sinusoidal untuk simulasi urban driving
    target_cycle = 30 + 15 * np.sin(0.1*t) + 8 * np.sin(0.5*t) + 5 * np.sin(1.5*t)
    target_cycle = np.clip(target_cycle, 0, 60)  # Batasi untuk urban driving
    
    # Tambahkan beberapa stops
    for i in range(len(target_cycle)):
        if i % 50 == 25:  # Stop setiap ~50 detik
            target_cycle[i:i+10] = 0
    
    # Inisialisasi optimizer
    optimizer = DrivingCycleOptimizer(target_cycle, duration)
    
    # Jalankan optimasi
    result, hba = optimizer.optimize_cycle(
        population_size=20,
        max_iter=30
    )
    
    # Visualisasi hasil
    optimized_cycle = result['best_position']
    
    plt.figure(figsize=(15, 10))
    
    # Plot comparison
    plt.subplot(2, 1, 1)
    time_minutes = np.arange(duration) / 60
    plt.plot(time_minutes, target_cycle, 'b-', linewidth=2, label='Target Cycle')
    plt.plot(time_minutes, optimized_cycle, 'r--', linewidth=2, label='Optimized Cycle')
    plt.title('Driving Cycle Optimization dengan HBA', fontsize=14, fontweight='bold')
    plt.ylabel('Speed (km/h)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot error
    plt.subplot(2, 1, 2)
    error = optimized_cycle - target_cycle
    plt.plot(time_minutes, error, 'g-', linewidth=1, label='Error')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Error (km/h)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    mse = np.mean(error**2)
    max_error = np.max(np.abs(error))
    print(f"\n📊 Optimasi Results:")
    print(f"   MSE: {mse:.4f}")
    print(f"   Max Error: {max_error:.4f} km/h")
    print(f"   Fitness Final: {result['best_fitness']:.4f}")
    
    return result, target_cycle, optimized_cycle

# ==================== ADVANCED HBA VARIANTS ====================

class ImprovedHBA(HoneyBadgerAlgorithm):
    """Improved HBA dengan enhancements"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_beta = True
        self.levy_flight = True
    
    def update_positions(self, iter: int):
        """Improved position update dengan adaptive parameters dan Levy flight"""
        I = self.calculate_intensity(self.fitness)
        density = self.update_density_factor(iter)
        
        # Adaptive beta
        if self.adaptive_beta:
            current_beta = self.beta * (1 - iter / self.max_iter)
        else:
            current_beta = self.beta
        
        new_positions = np.zeros_like(self.positions)
        
        for i in range(self.pop_size):
            F = np.random.choice([-1, 1])
            r1, r2, r3, r4 = np.random.random(4)
            
            # Levy flight component untuk meningkatkan eksplorasi
            if self.levy_flight and r4 < 0.3:
                levy_component = self.levy_flight_step(self.dim)
                digging_component = F * current_beta * I[i] * self.best_position + levy_component
            else:
                digging_component = F * current_beta * I[i] * self.best_position
            
            digging_component += F * r1 * density * self.positions[i]
            digging_component -= r2 * self.best_position
            
            honey_component = r3 * density * (self.positions[i] - self.best_position)
            
            if r4 < 0.5:
                new_pos = self.best_position + digging_component + honey_component
            else:
                new_pos = self.best_position + honey_component
            
            # Apply bounds
            for d in range(self.dim):
                new_pos[d] = np.clip(new_pos[d], self.bounds[d, 0], self.bounds[d, 1])
            
            new_positions[i] = new_pos
        
        # Evaluasi dan update
        new_fitness = np.array([self.objective_func(pos) for pos in new_positions])
        
        for i in range(self.pop_size):
            if new_fitness[i] < self.fitness[i]:
                self.positions[i] = new_positions[i]
                self.fitness[i] = new_fitness[i]
        
        # Update best solution
        current_best_idx = np.argmin(self.fitness)
        current_best_fitness = self.fitness[current_best_idx]
        
        if current_best_fitness < self.best_fitness:
            self.best_position = self.positions[current_best_idx].copy()
            self.best_fitness = current_best_fitness
        
        self.convergence_curve.append(self.best_fitness)
        self.position_history.append(self.positions.copy())
        self.fitness_history.append(self.fitness.copy())
    
    def levy_flight_step(self, dim: int) -> np.ndarray:
        """Generate Levy flight step"""
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        
        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        step = u / (np.abs(v) ** (1 / beta))
        
        return step

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("🐾 HONEY BADGER ALGORITHM IMPLEMENTATION")
    print("=" * 60)
    
    # Pilih demo yang ingin dijalankan
    print("Pilih demo:")
    print("1. Benchmark Functions")
    print("2. Driving Cycle Optimization")
    print("3. Improved HBA vs Standard HBA")
    
    choice = input("Masukkan pilihan (1-3): ").strip()
    
    if choice == "1":
        # Demo benchmark functions
        results = demo_benchmark_functions()
        
        # Generate report
        report_dfs = []
        for func_name, result in results.items():
            report_df = HBAAnalyzer.generate_report(result, func_name)
            report_dfs.append(report_df)
        
        final_report = pd.concat(report_dfs, ignore_index=True)
        print("\n" + "="*60)
        print("📈 PERFORMANCE REPORT")
        print("="*60)
        print(final_report.to_string(index=False))
    
    elif choice == "2":
        # Demo driving cycle optimization
        result, target, optimized = demo_driving_cycle_optimization()
    
    elif choice == "3":
        # Compare standard HBA vs improved HBA
        print("\n🔬 COMPARISON: Standard HBA vs Improved HBA")
        
        dim = 20
        bounds = [(-5.12, 5.12) for _ in range(dim)]
        
        # Standard HBA
        standard_hba = HoneyBadgerAlgorithm(
            objective_func=BenchmarkFunctions.rastrigin,
            dim=dim,
            bounds=bounds,
            population_size=30,
            max_iter=50,
            seed=42
        )
        standard_result = standard_hba.optimize()
        
        # Improved HBA
        improved_hba = ImprovedHBA(
            objective_func=BenchmarkFunctions.rastrigin,
            dim=dim,
            bounds=bounds,
            population_size=30,
            max_iter=50,
            seed=42
        )
        improved_result = improved_hba.optimize()
        
        # Comparison plot
        comparison_results = {
            'Standard HBA': standard_result,
            'Improved HBA': improved_result
        }
        
        HBAAnalyzer.compare_algorithms(comparison_results, "Standard HBA vs Improved HBA")
        
        # Print comparison
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"   Standard HBA Final Fitness: {standard_result['best_fitness']:.6f}")
        print(f"   Improved HBA Final Fitness: {improved_result['best_fitness']:.6f}")
        print(f"   Improvement: {((standard_result['best_fitness'] - improved_result['best_fitness']) / standard_result['best_fitness'] * 100):.2f}%")
    
    else:
        print("Pilihan tidak valid!")