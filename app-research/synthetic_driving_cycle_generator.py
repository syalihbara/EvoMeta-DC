# file: synthetic_driving_cycle_generator_individual.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators import compute_indicators
import os
import json

class DrivingCycleGenerator:
    def __init__(self, duration=600, dt=1.0):
        """
        Generator siklus mengemudi sintetik
        duration: durasi siklus (detik)
        dt: timestep (detik)
        """
        self.duration = duration
        self.dt = dt
        self.time = np.arange(0, duration, dt)
        self.n_points = len(self.time)
        
        # Parameter kendaraan default
        self.vehicle_params = {
            "mass": 1500,
            "Crr": 0.015,
            "Cd": 0.29,
            "A": 2.2,
            "rho": 1.225, 
            # Parameter baterai baru
            "battery_capacity": 50,  # kWh
            "motor_efficiency": 0.85,
            "regen_efficiency": 0.70,
            "auxiliary_power": 0.5,  # kW
            "initial_soc": 100  # %
        }
    
    def generate_hilly_cycle(self, cycle_id):
        """Generate driving cycle untuk domain hilly"""
        # Pattern kecepatan dasar dengan variasi untuk tanjakan/turunan
        base_speed = 15 + 3 * np.sin(2 * np.pi * self.time / 200)  # Komponen frekuensi rendah
        
        # Noise acak
        random_noise = np.random.normal(0, 2, self.n_points)
        
        # Pattern grade (kemiringan) yang signifikan
        grade = 0.05 * np.sin(2 * np.pi * self.time / 150) + 0.03 * np.sin(2 * np.pi * self.time / 80)
        
        # Pengaruh grade terhadap kecepatan
        grade_effect = -20 * np.abs(grade)  # Reduce speed on steep grades
        
        v = np.maximum(5, base_speed + random_noise + grade_effect)
        
        scenario_meta = {
            "grade": grade,
            "road_type": "hilly",
            "traffic_density": 0.3,
            "temperature": np.random.uniform(15, 25),
            "weather": np.random.choice(["clear", "cloudy"], p=[0.7, 0.3])
        }
        
        return v, scenario_meta
    
    def generate_urban_cycle(self, cycle_id):
        """Generate driving cycle untuk domain urban"""
        v = np.zeros(self.n_points)
        current_speed = 0
        
        for i in range(self.n_points):
            # Pattern urban: banyak stop-and-go
            if current_speed < 0.5:  # Berhenti
                if np.random.random() < 0.1:  # Mulai bergerak
                    current_speed = np.random.uniform(5, 15)
            else:  # Sedang bergerak
                if np.random.random() < 0.05:  # Berhenti di lampu merah
                    current_speed = 0
                else:  # Pertahankan kecepatan dengan variasi kecil
                    current_speed += np.random.normal(0, 1)
                    current_speed = np.clip(current_speed, 0, 50/3.6)  # Maks 50 km/jam
            
            v[i] = current_speed
        
        # Smooth the velocity profile
        v_series = pd.Series(v).rolling(window=5, center=True).mean()
        v_series = v_series.bfill().ffill()  # Ganti fillna(method) dengan bfill() dan ffill()
        v = v_series.values
        
        scenario_meta = {
            "grade": np.random.normal(0, 0.01, self.n_points),  # Jalan relatif datar
            "road_type": "urban",
            "traffic_density": np.random.uniform(0.6, 0.9),
            "temperature": np.random.uniform(20, 30),
            "weather": np.random.choice(["clear", "cloudy", "rainy"], p=[0.6, 0.3, 0.1])
        }
        
        return v, scenario_meta
    
    def generate_suburban_cycle(self, cycle_id):
        """Generate driving cycle untuk domain suburban"""
        # Pattern dengan akselerasi dan deselerasi moderat
        base_pattern = 20 + 8 * np.sin(2 * np.pi * self.time / 300)
        random_variation = np.random.normal(0, 3, self.n_points)
        
        v = np.maximum(0, base_pattern + random_variation)
        
        # Tambahkan beberapa periode kecepatan konstan
        cruise_segments = np.random.choice([True, False], self.n_points, p=[0.3, 0.7])
        cruise_mask = pd.Series(cruise_segments).rolling(10).mean() > 0.5
        v[cruise_mask] = np.random.uniform(40/3.6, 60/3.6)  # 40-60 km/jam
        
        scenario_meta = {
            "grade": np.random.normal(0, 0.02, self.n_points),
            "road_type": "suburban",
            "traffic_density": np.random.uniform(0.3, 0.6),
            "temperature": np.random.uniform(18, 28),
            "weather": np.random.choice(["clear", "cloudy"], p=[0.8, 0.2])
        }
        
        return v, scenario_meta
    
    def generate_congested_cycle(self, cycle_id):
        """Generate driving cycle untuk domain congested"""
        v = np.zeros(self.n_points)
        
        # Pattern kemacetan: kecepatan sangat rendah dengan banyak berhenti
        for i in range(self.n_points):
            if i % np.random.randint(20, 50) == 0:  # Bergerak sesekali
                v[i:i+np.random.randint(5, 15)] = np.random.uniform(2, 15/3.6)  # 2-15 km/jam
        
        # Smooth dan tambahkan noise
        v_series = pd.Series(v).rolling(window=3, center=True).mean()
        v_series = v_series.bfill().ffill().fillna(0)
        v = v_series.values
        v += np.random.normal(0, 0.5, self.n_points)
        v = np.maximum(0, v)
        
        scenario_meta = {
            "grade": np.random.normal(0, 0.005, self.n_points),
            "road_type": "congested",
            "traffic_density": np.random.uniform(0.8, 1.0),
            "temperature": np.random.uniform(22, 32),
            "weather": np.random.choice(["clear", "cloudy", "rainy"], p=[0.5, 0.3, 0.2])
        }
        
        return v, scenario_meta
    
    def generate_highway_cycle(self, cycle_id):
        """Generate driving cycle untuk domain highway"""
        # Kecepatan tinggi dengan variasi kecil
        base_speed = np.random.uniform(80/3.6, 100/3.6)  # 80-100 km/jam
        v = base_speed + np.random.normal(0, 2, self.n_points)
        
        # Simulasikan perubahan jalur atau menyalip
        for i in range(0, self.n_points, 100):
            if np.random.random() < 0.3:  # 30% chance of lane change/overtaking
                duration = np.random.randint(10, 30)
                speed_change = np.random.uniform(-5, 5)
                v[i:i+duration] += speed_change
        
        v = np.clip(v, 60/3.6, 120/3.6)  # Batasi 60-120 km/jam
        
        scenario_meta = {
            "grade": np.random.normal(0, 0.015, self.n_points),
            "road_type": "highway",
            "traffic_density": np.random.uniform(0.1, 0.4),
            "temperature": np.random.uniform(15, 25),
            "weather": np.random.choice(["clear", "cloudy"], p=[0.9, 0.1])
        }
        
        return v, scenario_meta
    
    def generate_dataset_individual_files(self, samples_per_domain=100, output_dir="driving_cycles"):
        """Generate dataset dengan analisis baterai"""
        domains = ["hilly", "urban", "suburban", "congested", "highway"]
        domain_generators = {
            "hilly": self.generate_hilly_cycle,
            "urban": self.generate_urban_cycle,
            "suburban": self.generate_suburban_cycle,
            "congested": self.generate_congested_cycle,
            "highway": self.generate_highway_cycle
        }
        
        os.makedirs(output_dir, exist_ok=True)
        all_metadata = []
        
        for domain in domains:
            print(f"Generating {samples_per_domain} samples for {domain} domain...")
            
            domain_dir = os.path.join(output_dir, domain)
            os.makedirs(domain_dir, exist_ok=True)
            
            for i in range(samples_per_domain):
                v, scenario_meta = domain_generators[domain](i)
                
                try:
                    indicators = compute_indicators(v, self.dt, self.vehicle_params, scenario_meta)
                    cycle_id = f"{domain}_{i:03d}"
                    indicators['cycle_id'] = cycle_id
                    indicators['domain'] = domain
                    
                    # Simpan dengan data baterai tambahan
                    self.save_individual_cycle(domain_dir, cycle_id, v, scenario_meta, indicators)
                    
                    all_metadata.append(indicators)
                    
                except Exception as e:
                    print(f"Error computing indicators for {domain}_{i}: {e}")
                    continue
        
        # Simpan metadata
        metadata_df = pd.DataFrame(all_metadata)
        metadata_path = os.path.join(output_dir, "metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Metadata saved to {metadata_path}")
        
        return metadata_df
    
    def save_individual_cycle(self, domain_dir, cycle_id, velocity, scenario_meta, indicators):
        """Simpan individual driving cycle dengan data baterai"""
        # Buat DataFrame untuk time series data + data baterai
        time_series_df = pd.DataFrame({
            'time': self.time,
            'velocity': velocity,
            'acceleration': np.gradient(velocity, self.dt),
            'grade': scenario_meta.get('grade', np.zeros_like(velocity))
        })
        
        # Simpan time series data
        csv_path = os.path.join(domain_dir, f"{cycle_id}.csv")
        time_series_df.to_csv(csv_path, index=False)
        
        # Simpan metadata dengan indikator baterai
        json_path = os.path.join(domain_dir, f"{cycle_id}_metadata.json")
        with open(json_path, 'w') as f:
            json_indicators = {}
            for key, value in indicators.items():
                if isinstance(value, (np.integer, np.int64)):
                    json_indicators[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    json_indicators[key] = float(value)
                else:
                    json_indicators[key] = value
            json.dump(json_indicators, f, indent=2)
        
        return csv_path, json_path
    
    def plot_sample_cycles(self, samples_per_domain=2):
        """Plot sample cycles dari setiap domain untuk visualisasi"""
        domains = ["hilly", "urban", "suburban", "congested", "highway"]
        domain_generators = {
            "hilly": self.generate_hilly_cycle,
            "urban": self.generate_urban_cycle,
            "suburban": self.generate_suburban_cycle,
            "congested": self.generate_congested_cycle,
            "highway": self.generate_highway_cycle
        }
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, domain in enumerate(domains):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            for i in range(samples_per_domain):
                v, scenario_meta = domain_generators[domain](i)
                ax.plot(self.time, v * 3.6, label=f'Sample {i+1}', alpha=0.7)  # Convert to km/h
                
            ax.set_title(f'{domain.capitalize()} Domain')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Speed (km/h)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(domains), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('driving_cycle_samples.png', dpi=300, bbox_inches='tight')
        plt.show()

def analyze_dataset(output_dir="driving_cycles"):
    """Analisis statistik dataset yang dihasilkan"""
    metadata_path = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_path):
        print("Metadata file not found. Please generate dataset first.")
        return
    
    df = pd.read_csv(metadata_path)
    
    print("Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print("\nSamples per domain:")
    print(df['domain'].value_counts())
    
    print("\nStatistical summary for key indicators:")
    key_indicators = ['avg_speed', 'max_speed', 'ev_Wh_per_km', 'peak_power_kW', 
                     'idle_ratio', 'accel_ratio', 'decel_ratio']
    
    print(df[key_indicators].describe())
    
    # Group by domain untuk melihat perbedaan karakteristik
    domain_stats = df.groupby('domain')[key_indicators].mean()
    print("\nAverage indicators by domain:")
    print(domain_stats)
    
    # Count files per domain
    print("\nFiles generated per domain:")
    for domain in df['domain'].unique():
        domain_dir = os.path.join(output_dir, domain)
        if os.path.exists(domain_dir):
            file_count = len([f for f in os.listdir(domain_dir) if f.endswith('.csv')])
            print(f"{domain}: {file_count} files")

if __name__ == "__main__":
    # Initialize generator
    generator = DrivingCycleGenerator(duration=600, dt=1.0)  # 10 menit cycles
    
    # Plot sample cycles untuk inspeksi visual
    print("Generating sample cycles for visualization...")
    generator.plot_sample_cycles()
    
    # Generate full dataset dengan file individual
    print("\nGenerating full dataset with individual files...")
    output_directory = "driving_cycles_dataset"
    df = generator.generate_dataset_individual_files(
        samples_per_domain=100,  # 100 samples per domain
        output_dir=output_directory
    )
    
    # Analyze the dataset
    print("\nAnalyzing generated dataset...")
    analyze_dataset(output_directory)
    
    print("\nDataset generation completed!")
    print(f"Generated {len(df)} samples across 5 domains")
    print(f"Files saved in: {output_directory}")