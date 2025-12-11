import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import ks_2samp, wasserstein_distance
import random
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveEVDrivingCycleGenerator:
    """
    Aplikasi komprehensif untuk generating dataset sintetis driving cycle 
    untuk kendaraan listrik dengan 5 domain dan validasi terhadap standar uji
    """
    
    def __init__(self):
        self.standards = self._define_standard_parameters()
        self.domains = self._define_domain_parameters()
        self.methods = ['markov_chain', 'fourier', 'statistical']
        
    def _define_standard_parameters(self) -> Dict:
        """Mendefinisikan parameter statistik dari standar uji existing"""
        return {
            'WLTP': {
                'v_avg': (46.5, 2.0), 'v_max': (131, 5.0), 'v_std': (30.2, 2.0),
                'a_avg': (0.45, 0.05), 'a_std': (0.85, 0.1), 'idle_ratio': (0.13, 0.02),
                'pos_accel_ratio': (0.28, 0.03), 'neg_accel_ratio': (0.25, 0.03),
                'cruise_ratio': (0.34, 0.03), 'stop_frequency': (0.8, 0.1)
            },
            'EPA': {
                'v_avg': (34.1, 3.0), 'v_max': (91.2, 4.0), 'v_std': (25.8, 2.0),
                'a_avg': (0.50, 0.06), 'a_std': (0.92, 0.1), 'idle_ratio': (0.18, 0.03),
                'pos_accel_ratio': (0.25, 0.03), 'neg_accel_ratio': (0.23, 0.03),
                'cruise_ratio': (0.34, 0.03), 'stop_frequency': (1.2, 0.2)
            },
            'CLTC': {
                'v_avg': (28.9, 2.5), 'v_max': (114, 4.0), 'v_std': (22.4, 2.0),
                'a_avg': (0.42, 0.05), 'a_std': (0.78, 0.1), 'idle_ratio': (0.22, 0.03),
                'pos_accel_ratio': (0.26, 0.03), 'neg_accel_ratio': (0.24, 0.03),
                'cruise_ratio': (0.28, 0.03), 'stop_frequency': (1.5, 0.3)
            }
        }
    
    def _define_domain_parameters(self) -> Dict:
        """Mendefinisikan karakteristik untuk setiap domain"""
        return {
            'urban': {
                'v_avg_range': (15, 35), 'v_max_range': (60, 80), 'idle_range': (0.15, 0.25),
                'accel_intensity': 'medium', 'stop_frequency': 'high',
                'description': 'Kondisi perkotaan dengan banyak traffic light dan persimpangan'
            },
            'suburban': {
                'v_avg_range': (35, 55), 'v_max_range': (70, 90), 'idle_range': (0.08, 0.15),
                'accel_intensity': 'medium', 'stop_frequency': 'medium',
                'description': 'Kondisi pinggiran kota dengan jalan arteri'
            },
            'highway': {
                'v_avg_range': (70, 90), 'v_max_range': (100, 130), 'idle_range': (0.01, 0.05),
                'accel_intensity': 'low', 'stop_frequency': 'low',
                'description': 'Jalan tol dan jalan bebas hambatan'
            },
            'congested': {
                'v_avg_range': (5, 20), 'v_max_range': (40, 60), 'idle_range': (0.25, 0.40),
                'accel_intensity': 'high', 'stop_frequency': 'very_high',
                'description': 'Kemacetan parah dengan banyak stop-and-go'
            },
            'hilly': {
                'v_avg_range': (25, 45), 'v_max_range': (70, 90), 'idle_range': (0.10, 0.20),
                'accel_intensity': 'high', 'stop_frequency': 'medium',
                'gradient_range': (-8, 8), 'elevation_gain_range': (200, 600),
                'description': 'Medan berbukit dengan tanjakan dan turunan signifikan'
            }
        }

    def generate_driving_cycle(self, domain: str, method: str, duration: int = 1800, 
                             cycle_id: str = None) -> Dict:
        """Generate driving cycle berdasarkan domain dan metode yang dipilih"""
        
        if method == 'markov_chain':
            cycle = self._markov_chain_method(domain, duration)
        elif method == 'fourier':
            cycle = self._fourier_method(domain, duration)
        elif method == 'statistical':
            cycle = self._statistical_method(domain, duration)
        else:
            raise ValueError(f"Method {method} tidak dikenali")
        
        # Tambahkan elevation profile
        cycle = self._add_elevation_profile(cycle, domain)
        
        # Hitung metrics
        metrics = self._calculate_cycle_metrics(cycle)
        
        # Validasi terhadap standar
        validations = {}
        for standard in self.standards.keys():
            validations[standard] = self._validate_against_standard(metrics, standard)
        
        # Siapkan output
        cycle_data = {
            'cycle_id': cycle_id or f"{domain}_{method}_{datetime.now().strftime('%H%M%S')}",
            'domain': domain,
            'method': method,
            'duration_s': duration,
            'timestamp': datetime.now().isoformat(),
            'data': cycle,
            'metrics': metrics,
            'validations': validations
        }
        
        return cycle_data

    def _markov_chain_method(self, domain: str, duration: int = 1800) -> Dict:
        """Metode Markov Chain untuk generating driving cycle"""
        
        state_params = self._get_markov_states(domain)
        states = state_params['states']
        transition_matrix = state_params['transition_matrix']
        speed_ranges = state_params['speed_ranges']
        
        current_state = 0
        speeds = [0]
        accelerations = [0]
        states_history = [current_state]
        
        for t in range(1, duration):
            next_state = np.random.choice(
                len(states), 
                p=transition_matrix[current_state]
            )
            
            speed_range = speed_ranges[states[next_state][0]]
            if speed_range[0] == 0 and speed_range[1] == 0:
                new_speed = 0
            else:
                base_speed = np.random.uniform(speed_range[0], speed_range[1])
                max_accel = 3.0 if states[next_state][1] == 'accel' else 2.5
                speed_diff = base_speed - speeds[-1]
                constrained_diff = np.clip(speed_diff, -max_accel * 3.6, max_accel * 3.6)
                new_speed = speeds[-1] + constrained_diff
                new_speed = max(0, min(130, new_speed))
            
            speeds.append(new_speed)
            accel = (speeds[t] - speeds[t-1]) / 3.6
            accelerations.append(accel)
            states_history.append(next_state)
            current_state = next_state
        
        return {
            'time': list(range(duration)),
            'speed_kmh': speeds,
            'acceleration_mss': accelerations,
            'domain': domain,
            'method': 'markov_chain'
        }

    def _get_markov_states(self, domain: str) -> Dict:
        """Mendapatkan parameter Markov Chain berdasarkan domain"""
        
        if domain == 'urban':
            return {
                'states': [
                    ('idle', 'zero'), ('very_low', 'decel'), ('very_low', 'accel'),
                    ('low', 'cruise'), ('low', 'accel'), ('medium', 'cruise')
                ],
                'speed_ranges': {
                    'idle': (0, 0), 'very_low': (1, 15), 'low': (16, 35), 
                    'medium': (36, 50)
                },
                'transition_matrix': np.array([
                    [0.1, 0.4, 0.3, 0.1, 0.1, 0.0],
                    [0.3, 0.2, 0.2, 0.2, 0.1, 0.0],
                    [0.1, 0.2, 0.2, 0.3, 0.2, 0.0],
                    [0.05, 0.2, 0.1, 0.3, 0.2, 0.15],
                    [0.0, 0.1, 0.1, 0.2, 0.3, 0.3],
                    [0.0, 0.05, 0.0, 0.2, 0.2, 0.55]
                ])
            }
        elif domain == 'highway':
            return {
                'states': [
                    ('medium', 'cruise'), ('high', 'cruise'), ('very_high', 'cruise'),
                    ('high', 'accel'), ('high', 'decel'), ('medium', 'decel')
                ],
                'speed_ranges': {
                    'medium': (50, 70), 'high': (71, 100), 'very_high': (101, 130)
                },
                'transition_matrix': np.array([
                    [0.4, 0.3, 0.1, 0.1, 0.1, 0.0],
                    [0.2, 0.5, 0.1, 0.1, 0.1, 0.0],
                    [0.1, 0.3, 0.4, 0.1, 0.1, 0.0],
                    [0.1, 0.2, 0.1, 0.4, 0.2, 0.0],
                    [0.1, 0.2, 0.1, 0.2, 0.4, 0.0],
                    [0.3, 0.2, 0.0, 0.1, 0.2, 0.2]
                ])
            }
        elif domain == 'congested':
            return {
                'states': [
                    ('idle', 'zero'), ('idle', 'zero'), ('very_low', 'decel'),
                    ('very_low', 'accel'), ('very_low', 'cruise'), ('idle', 'zero')
                ],
                'speed_ranges': {
                    'idle': (0, 0), 'very_low': (1, 20)
                },
                'transition_matrix': np.array([
                    [0.6, 0.2, 0.1, 0.05, 0.05, 0.0],
                    [0.3, 0.4, 0.1, 0.1, 0.1, 0.0],
                    [0.2, 0.1, 0.3, 0.2, 0.2, 0.0],
                    [0.1, 0.1, 0.2, 0.3, 0.3, 0.0],
                    [0.05, 0.05, 0.2, 0.3, 0.4, 0.0],
                    [0.4, 0.3, 0.1, 0.1, 0.1, 0.0]
                ])
            }
        elif domain == 'hilly':
            return {
                'states': [
                    ('low', 'accel'), ('medium', 'decel'), ('low', 'decel'),
                    ('medium', 'accel'), ('low', 'cruise'), ('medium', 'cruise')
                ],
                'speed_ranges': {
                    'low': (10, 40), 'medium': (41, 70)
                },
                'transition_matrix': np.array([
                    [0.2, 0.3, 0.1, 0.2, 0.1, 0.1],
                    [0.3, 0.2, 0.2, 0.1, 0.1, 0.1],
                    [0.1, 0.2, 0.3, 0.1, 0.2, 0.1],
                    [0.2, 0.1, 0.1, 0.3, 0.1, 0.2],
                    [0.1, 0.1, 0.2, 0.1, 0.3, 0.2],
                    [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]
                ])
            }
        else:  # suburban
            return {
                'states': [
                    ('idle', 'zero'), ('low', 'decel'), ('low', 'accel'),
                    ('medium', 'cruise'), ('medium', 'accel'), ('high', 'cruise')
                ],
                'speed_ranges': {
                    'idle': (0, 0), 'low': (1, 40), 'medium': (41, 70), 
                    'high': (71, 90)
                },
                'transition_matrix': np.array([
                    [0.1, 0.3, 0.3, 0.2, 0.1, 0.0],
                    [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
                    [0.1, 0.2, 0.2, 0.3, 0.2, 0.0],
                    [0.05, 0.15, 0.1, 0.4, 0.2, 0.1],
                    [0.0, 0.1, 0.1, 0.2, 0.4, 0.2],
                    [0.0, 0.05, 0.0, 0.2, 0.2, 0.55]
                ])
            }

    def _fourier_method(self, domain: str, duration: int = 1800) -> Dict:
        """Metode Fourier Series dengan domain-specific parameters"""
        
        domain_params = self.domains[domain]
        t = np.linspace(0, 4*np.pi, duration)
        
        base_speed = np.mean(domain_params['v_avg_range'])
        fourier_components = []
        components_count = random.randint(3, 7)
        
        for i in range(1, components_count + 1):
            if domain in ['urban', 'congested']:
                amplitude = random.uniform(3, 8) / i
                frequency = random.uniform(0.2, 0.8) * i
            elif domain == 'highway':
                amplitude = random.uniform(2, 5) / i  
                frequency = random.uniform(0.1, 0.3) * i
            else:
                amplitude = random.uniform(4, 10) / i
                frequency = random.uniform(0.15, 0.6) * i
                
            phase = random.uniform(0, 2*np.pi)
            component = amplitude * np.sin(frequency * t + phase)
            fourier_components.append(component)
        
        speed_variation = sum(fourier_components)
        
        if domain in ['urban', 'congested']:
            noise = np.random.normal(0, 2, duration)
        else:
            noise = np.random.normal(0, 1.5, duration)
        
        speeds = base_speed + speed_variation + noise
        speeds = np.maximum(0, speeds)
        speeds = np.minimum(domain_params['v_max_range'][1] * 1.1, speeds)
        
        if domain == 'highway':
            window = 71
        else:
            window = 51
        speeds = signal.savgol_filter(speeds, window_length=window, polyorder=3)
        
        if domain in ['urban', 'congested']:
            idle_prob = domain_params['idle_range'][1] / 100
            for i in range(len(speeds)):
                if random.random() < idle_prob/10:
                    idle_duration = random.randint(15, 45) if domain == 'congested' else random.randint(10, 30)
                    speeds[i:i+idle_duration] = 0
        
        accelerations = [0]
        for i in range(1, len(speeds)):
            accel = (speeds[i] - speeds[i-1]) / 3.6
            accelerations.append(accel)
        
        return {
            'time': list(range(duration)),
            'speed_kmh': speeds.tolist(),
            'acceleration_mss': accelerations,
            'domain': domain,
            'method': 'fourier'
        }

    def _statistical_method(self, domain: str, duration: int = 1800) -> Dict:
        """Metode Statistical Distribution berdasarkan domain"""
        
        domain_params = self.domains[domain]
        
        if domain == 'urban':
            speed_mean = np.mean(domain_params['v_avg_range'])
            speed_std = speed_mean * 0.4
            idle_prob = 0.18
        elif domain == 'highway':
            speed_mean = np.mean(domain_params['v_avg_range'])
            speed_std = speed_mean * 0.2
            idle_prob = 0.02
        elif domain == 'congested':
            speed_mean = np.mean(domain_params['v_avg_range'])
            speed_std = speed_mean * 0.6
            idle_prob = 0.30
        elif domain == 'hilly':
            speed_mean = np.mean(domain_params['v_avg_range'])
            speed_std = speed_mean * 0.35
            idle_prob = 0.12
        else:
            speed_mean = np.mean(domain_params['v_avg_range'])
            speed_std = speed_mean * 0.3
            idle_prob = 0.10
        
        speeds = [0]
        accelerations = [0]
        
        for t in range(1, duration):
            if random.random() < idle_prob:
                new_speed = 0
            else:
                new_speed = np.random.normal(speed_mean, speed_std)
                new_speed = max(0, min(domain_params['v_max_range'][1] * 1.1, new_speed))
                
                max_accel = 2.8
                speed_diff = new_speed - speeds[-1]
                constrained_diff = np.clip(speed_diff, -max_accel * 3.6, max_accel * 3.6)
                new_speed = speeds[-1] + constrained_diff
            
            speeds.append(new_speed)
            accel = (speeds[t] - speeds[t-1]) / 3.6
            accelerations.append(accel)
        
        return {
            'time': list(range(duration)),
            'speed_kmh': speeds,
            'acceleration_mss': accelerations,
            'domain': domain,
            'method': 'statistical'
        }

    def _add_elevation_profile(self, cycle: Dict, domain: str) -> Dict:
        """Menambahkan elevation profile khusus untuk domain hilly"""
        
        if domain != 'hilly':
            elevation = np.random.normal(0, 10, len(cycle['speed_kmh']))
            elevation = np.cumsum(elevation)
            elevation = elevation - elevation[0]
        else:
            duration = len(cycle['speed_kmh'])
            t = np.linspace(0, 4*np.pi, duration)
            elevation = 50 * np.sin(0.5 * t) + 30 * np.sin(1.5 * t + 1) + 20 * np.sin(3 * t + 2)
            elevation += np.random.normal(0, 15, duration)
            elevation = signal.savgol_filter(elevation, window_length=201, polyorder=3)
            
            min_elev, max_elev = self.domains['hilly']['elevation_gain_range']
            elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min()) * (max_elev - min_elev) + min_elev
        
        gradient = np.gradient(elevation)
        speeds_ms = np.array(cycle['speed_kmh']) / 3.6
        speeds_ms[speeds_ms < 0.1] = 0.1
        gradient_pct = (gradient / speeds_ms) * 100
        gradient_pct = np.clip(gradient_pct, -15, 15)
        
        cycle['elevation_m'] = elevation.tolist()
        cycle['gradient_pct'] = gradient_pct.tolist()
        
        return cycle

    def _calculate_cycle_metrics(self, cycle: Dict) -> Dict:
        """Menghitung metrics statistik untuk driving cycle"""
        
        speeds = np.array(cycle['speed_kmh'])
        accelerations = np.array(cycle['acceleration_mss'])
        
        total_time = len(speeds)
        idle_mask = speeds == 0
        moving_mask = ~idle_mask
        
        idle_time = np.sum(idle_mask)
        moving_time = total_time - idle_time
        
        distance_km = np.sum(speeds) / 3600
        
        pos_accel_mask = accelerations > 0.1
        neg_accel_mask = accelerations < -0.1
        cruise_mask = (accelerations >= -0.1) & (accelerations <= 0.1) & (speeds > 0)
        
        metrics = {
            'total_distance_km': distance_km,
            'average_speed_kmh': np.mean(speeds),
            'max_speed_kmh': np.max(speeds),
            'speed_std_kmh': np.std(speeds),
            'idle_ratio': idle_time / total_time,
            'moving_average_speed_kmh': np.mean(speeds[moving_mask]) if moving_time > 0 else 0,
            
            'average_acceleration_mss': np.mean(accelerations),
            'acceleration_std_mss': np.std(accelerations),
            'max_acceleration_mss': np.max(accelerations),
            'min_acceleration_mss': np.min(accelerations),
            
            'positive_accel_ratio': np.sum(pos_accel_mask) / total_time,
            'negative_accel_ratio': np.sum(neg_accel_mask) / total_time,
            'cruise_ratio': np.sum(cruise_mask) / total_time,
            
            'stop_count': self._count_stops(speeds),
            'average_stop_duration': self._average_stop_duration(speeds)
        }
        
        if 'elevation_m' in cycle:
            elevation = np.array(cycle['elevation_m'])
            metrics['elevation_gain_m'] = np.max(elevation) - np.min(elevation)
            metrics['max_gradient_pct'] = np.max(np.abs(cycle['gradient_pct']))
        
        return metrics

    def _count_stops(self, speeds: np.ndarray, min_stop_duration: int = 3) -> int:
        """Menghitung jumlah stop"""
        stops = 0
        current_stop_duration = 0
        
        for speed in speeds:
            if speed == 0:
                current_stop_duration += 1
            else:
                if current_stop_duration >= min_stop_duration:
                    stops += 1
                current_stop_duration = 0
        
        if current_stop_duration >= min_stop_duration:
            stops += 1
            
        return stops

    def _average_stop_duration(self, speeds: np.ndarray, min_stop_duration: int = 3) -> float:
        """Menghitung rata-rata durasi stop"""
        stop_durations = []
        current_stop_duration = 0
        
        for speed in speeds:
            if speed == 0:
                current_stop_duration += 1
            else:
                if current_stop_duration >= min_stop_duration:
                    stop_durations.append(current_stop_duration)
                current_stop_duration = 0
        
        if current_stop_duration >= min_stop_duration:
            stop_durations.append(current_stop_duration)
        
        return np.mean(stop_durations) if stop_durations else 0

    def _validate_against_standard(self, cycle_metrics: Dict, standard: str) -> Dict:
        """Validasi cycle terhadap standar uji"""
        
        if standard not in self.standards:
            raise ValueError(f"Standard {standard} tidak dikenali")
        
        standard_params = self.standards[standard]
        validation_results = {}
        
        for param, (target_mean, target_std) in standard_params.items():
            if param in cycle_metrics:
                cycle_value = cycle_metrics[param]
                z_score = abs(cycle_value - target_mean) / target_std
                similarity = max(0, 1 - z_score / 3)
                validation_results[param] = {
                    'cycle_value': cycle_value,
                    'target_value': target_mean,
                    'z_score': z_score,
                    'similarity_score': similarity
                }
        
        overall_similarity = np.mean([v['similarity_score'] for v in validation_results.values()])
        validation_results['overall_similarity'] = overall_similarity
        
        return validation_results

    def generate_dataset(self, cycles_per_domain: int = 10, duration: int = 1800) -> Dict:
        """Generate dataset komprehensif untuk semua domain dan metode"""
        
        dataset = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'total_cycles': cycles_per_domain * len(self.domains) * len(self.methods),
                'duration_per_cycle': duration,
                'domains': list(self.domains.keys()),
                'methods': self.methods
            },
            'cycles': []
        }
        
        for domain in self.domains.keys():
            print(f"Generating {cycles_per_domain} cycles for domain: {domain}")
            
            for method in self.methods:
                for i in range(cycles_per_domain):
                    cycle_id = f"{domain}_{method}_{i:03d}"
                    cycle_data = self.generate_driving_cycle(
                        domain=domain,
                        method=method,
                        duration=duration,
                        cycle_id=cycle_id
                    )
                    dataset['cycles'].append(cycle_data)
        
        return dataset

    def analyze_method_performance(self, dataset: Dict) -> pd.DataFrame:
        """Analisis performa setiap metode per domain dan standar"""
        
        results = []
        
        for cycle in dataset['cycles']:
            for standard, validation in cycle['validations'].items():
                results.append({
                    'cycle_id': cycle['cycle_id'],
                    'domain': cycle['domain'],
                    'method': cycle['method'],
                    'standard': standard,
                    'overall_similarity': validation['overall_similarity']
                })
        
        df = pd.DataFrame(results)
        
        # Analisis metode terbaik
        best_methods = df.groupby(['domain', 'standard', 'method'])['overall_similarity'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        return best_methods

    def get_recommended_methods(self, dataset: Dict) -> Dict:
        """Mendapatkan rekomendasi metode terbaik per domain"""
        
        analysis_df = self.analyze_method_performance(dataset)
        recommended_methods = {}
        
        for domain in self.domains.keys():
            domain_data = analysis_df[analysis_df['domain'] == domain]
            best_methods = {}
            
            for standard in self.standards.keys():
                standard_data = domain_data[domain_data['standard'] == standard]
                if not standard_data.empty:
                    best_idx = standard_data['mean'].idxmax()
                    best_method = standard_data.loc[best_idx, 'method']
                    best_score = standard_data.loc[best_idx, 'mean']
                    best_methods[standard] = {
                        'method': best_method,
                        'similarity_score': best_score
                    }
            
            # Overall recommendation
            overall_best = domain_data.groupby('method')['mean'].mean().idxmax()
            overall_score = domain_data.groupby('method')['mean'].mean().max()
            
            recommended_methods[domain] = {
                'standard_specific': best_methods,
                'overall_recommendation': {
                    'method': overall_best,
                    'average_similarity': overall_score
                }
            }
        
        return recommended_methods

    def create_comprehensive_report(self, dataset: Dict):
        """Membuat laporan komprehensif"""
        
        print("=" * 80)
        print("COMPREHENSIVE DRIVING CYCLE SYNTHETIC DATASET REPORT")
        print("=" * 80)
        
        # Metadata
        metadata = dataset['metadata']
        print(f"\nDATASET METADATA:")
        print(f"Generation Date: {metadata['generation_date']}")
        print(f"Total Cycles: {metadata['total_cycles']}")
        print(f"Duration per Cycle: {metadata['duration_per_cycle']} seconds")
        print(f"Domains: {', '.join(metadata['domains'])}")
        print(f"Methods: {', '.join(metadata['methods'])}")
        
        # Recommended methods
        recommendations = self.get_recommended_methods(dataset)
        
        print("\n" + "=" * 50)
        print("RECOMMENDED METHODS PER DOMAIN")
        print("=" * 50)
        
        for domain, rec_data in recommendations.items():
            overall_rec = rec_data['overall_recommendation']
            print(f"\n{domain.upper():<12}: {overall_rec['method']} "
                  f"(Avg Similarity: {overall_rec['average_similarity']:.3f})")
            print(f"Description: {self.domains[domain]['description']}")
            
            print("  Standard-specific recommendations:")
            for standard, spec_rec in rec_data['standard_specific'].items():
                print(f"    {standard}: {spec_rec['method']} "
                      f"(Score: {spec_rec['similarity_score']:.3f})")
        
        # Overall statistics
        print("\n" + "=" * 50)
        print("OVERALL STATISTICS")
        print("=" * 50)
        
        all_similarities = []
        for cycle in dataset['cycles']:
            for standard, validation in cycle['validations'].items():
                all_similarities.append(validation['overall_similarity'])
        
        print(f"Average Similarity: {np.mean(all_similarities):.3f} ± {np.std(all_similarities):.3f}")
        print(f"Similarity Range: [{np.min(all_similarities):.3f}, {np.max(all_similarities):.3f}]")
        
        # Method performance summary
        print("\n" + "=" * 50)
        print("METHOD PERFORMANCE SUMMARY")
        print("=" * 50)
        
        method_results = []
        for cycle in dataset['cycles']:
            avg_similarity = np.mean([v['overall_similarity'] for v in cycle['validations'].values()])
            method_results.append({
                'method': cycle['method'],
                'similarity': avg_similarity,
                'domain': cycle['domain']
            })
        
        method_df = pd.DataFrame(method_results)
        method_summary = method_df.groupby('method')['similarity'].agg(['mean', 'std', 'count'])
        print(method_summary.round(3))

    def export_dataset(self, dataset: Dict, base_filename: str = None):
        """Export dataset ke berbagai format"""
        
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"ev_driving_cycles_{timestamp}"
        
        # Export to Parquet (main data)
        cycles_data = []
        for cycle in dataset['cycles']:
            cycle_df = pd.DataFrame(cycle['data'])
            cycle_df['cycle_id'] = cycle['cycle_id']
            cycle_df['domain'] = cycle['domain']
            cycle_df['method'] = cycle['method']
            cycles_data.append(cycle_df)
        
        combined_data = pd.concat(cycles_data, ignore_index=True)
        combined_data.to_parquet(f'{base_filename}_data.parquet')
        
        # Export metrics summary
        metrics_data = []
        for cycle in dataset['cycles']:
            metrics = cycle['metrics'].copy()
            metrics['cycle_id'] = cycle['cycle_id']
            metrics['domain'] = cycle['domain']
            metrics['method'] = cycle['method']
            
            # Add validation scores
            for standard, validation in cycle['validations'].items():
                metrics[f'{standard}_similarity'] = validation['overall_similarity']
            
            metrics_data.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_parquet(f'{base_filename}_metrics.parquet')
        metrics_df.to_csv(f'{base_filename}_metrics.csv', index=False)
        
        # Export full dataset as JSON
        with open(f'{base_filename}_full.json', 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        print(f"\nDataset exported with base filename: {base_filename}")

    def visualize_dataset(self, dataset: Dict, save_path: str = None):
        """Visualisasi komprehensif dataset"""
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Similarity Analysis
        ax1 = plt.subplot(3, 3, 1)
        similarity_data = []
        for cycle in dataset['cycles']:
            for standard, validation in cycle['validations'].items():
                similarity_data.append({
                    'domain': cycle['domain'],
                    'method': cycle['method'],
                    'standard': standard,
                    'similarity': validation['overall_similarity']
                })
        
        similarity_df = pd.DataFrame(similarity_data)
        sns.boxplot(data=similarity_df, x='domain', y='similarity', hue='method', ax=ax1)
        ax1.set_title('Similarity Score by Domain and Method')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Method Performance Comparison
        ax2 = plt.subplot(3, 3, 2)
        method_performance = similarity_df.groupby(['method', 'standard'])['similarity'].mean().unstack()
        method_performance.plot(kind='bar', ax=ax2)
        ax2.set_title('Method Performance by Standard')
        ax2.set_ylabel('Average Similarity')
        ax2.legend(title='Standard')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Example Cycles
        ax3 = plt.subplot(3, 3, 3)
        domains = list(self.domains.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(domains)))
        
        for i, domain in enumerate(domains):
            domain_cycles = [c for c in dataset['cycles'] if c['domain'] == domain]
            if domain_cycles:
                cycle = domain_cycles[0]['data']
                time = np.array(cycle['time']) / 60
                ax3.plot(time[:600], cycle['speed_kmh'][:600], 
                        color=colors[i], label=domain, linewidth=1.5, alpha=0.8)
        
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Speed (km/h)')
        ax3.set_title('Example Driving Cycles\n(First 10 minutes)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Speed Distribution
        ax4 = plt.subplot(3, 3, 4)
        speed_data = []
        for cycle in dataset['cycles']:
            speeds = cycle['data']['speed_kmh']
            domain = cycle['domain']
            speed_data.extend([(speed, domain) for speed in speeds if speed > 0])
        
        speed_df = pd.DataFrame(speed_data, columns=['speed', 'domain'])
        for domain in domains:
            domain_speeds = speed_df[speed_df['domain'] == domain]['speed']
            if len(domain_speeds) > 0:
                ax4.hist(domain_speeds, bins=50, alpha=0.6, label=domain, density=True)
        
        ax4.set_xlabel('Speed (km/h)')
        ax4.set_ylabel('Density')
        ax4.set_title('Speed Distribution by Domain')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Acceleration Distribution
        ax5 = plt.subplot(3, 3, 5)
        accel_data = []
        for cycle in dataset['cycles']:
            accelerations = cycle['data']['acceleration_mss']
            domain = cycle['domain']
            accel_data.extend([(accel, domain) for accel in accelerations])
        
        accel_df = pd.DataFrame(accel_data, columns=['acceleration', 'domain'])
        for domain in domains:
            domain_accels = accel_df[accel_df['domain'] == domain]['acceleration']
            if len(domain_accels) > 0:
                ax5.hist(domain_accels, bins=50, alpha=0.6, label=domain, density=True)
        
        ax5.set_xlabel('Acceleration (m/s²)')
        ax5.set_ylabel('Density')
        ax5.set_title('Acceleration Distribution by Domain')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(-3, 3)
        
        # 6. Idle Ratio by Domain
        ax6 = plt.subplot(3, 3, 6)
        idle_data = []
        for cycle in dataset['cycles']:
            idle_data.append({
                'domain': cycle['domain'],
                'idle_ratio': cycle['metrics']['idle_ratio']
            })
        
        idle_df = pd.DataFrame(idle_data)
        sns.boxplot(data=idle_df, x='domain', y='idle_ratio', ax=ax6)
        ax6.set_title('Idle Ratio Distribution by Domain')
        ax6.set_ylabel('Idle Ratio')
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. Elevation Profile Example (Hilly domain)
        ax7 = plt.subplot(3, 3, 7)
        hilly_cycles = [c for c in dataset['cycles'] if c['domain'] == 'hilly']
        if hilly_cycles:
            cycle = hilly_cycles[0]['data']
            time = np.array(cycle['time']) / 60
            ax7.plot(time, cycle['speed_kmh'], 'b-', alpha=0.7, label='Speed')
            ax7_twin = ax7.twinx()
            ax7_twin.plot(time, cycle['elevation_m'], 'r-', alpha=0.7, label='Elevation')
            ax7.set_xlabel('Time (minutes)')
            ax7.set_ylabel('Speed (km/h)', color='b')
            ax7_twin.set_ylabel('Elevation (m)', color='r')
            ax7.set_title('Hilly Domain: Speed vs Elevation')
            ax7.legend(loc='upper left')
            ax7_twin.legend(loc='upper right')
            ax7.grid(True, alpha=0.3)
        
        # 8. Overall Similarity Heatmap
        ax8 = plt.subplot(3, 3, 8)
        heatmap_data = []
        for domain in domains:
            for method in self.methods:
                domain_method_cycles = [c for c in dataset['cycles'] 
                                      if c['domain'] == domain and c['method'] == method]
                if domain_method_cycles:
                    similarities = []
                    for cycle in domain_method_cycles:
                        avg_similarity = np.mean([v['overall_similarity'] 
                                                for v in cycle['validations'].values()])
                        similarities.append(avg_similarity)
                    heatmap_data.append({
                        'domain': domain,
                        'method': method,
                        'similarity': np.mean(similarities)
                    })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_pivot = heatmap_df.pivot(index='domain', columns='method', values='similarity')
        sns.heatmap(heatmap_pivot, annot=True, cmap='YlOrRd', ax=ax8, cbar_kws={'label': 'Similarity Score'})
        ax8.set_title('Average Similarity Heatmap')
        
        # 9. Standard Comparison
        ax9 = plt.subplot(3, 3, 9)
        standard_data = []
        for cycle in dataset['cycles']:
            for standard, validation in cycle['validations'].items():
                standard_data.append({
                    'standard': standard,
                    'similarity': validation['overall_similarity'],
                    'domain': cycle['domain']
                })
        
        standard_df = pd.DataFrame(standard_data)
        sns.boxplot(data=standard_df, x='standard', y='similarity', ax=ax9)
        ax9.set_title('Similarity Distribution by Standard')
        ax9.set_ylabel('Similarity Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()

# Contoh penggunaan aplikasi
def main():
    """Fungsi utama untuk menjalankan aplikasi"""
    
    print("🚀 Starting Comprehensive EV Driving Cycle Generator")
    print("=" * 60)
    
    # Inisialisasi generator
    generator = ComprehensiveEVDrivingCycleGenerator()
    
    # Generate dataset
    print("\n📊 Generating synthetic driving cycle dataset...")
    dataset = generator.generate_dataset(
        cycles_per_domain=8,  # 8 cycles per domain per method
        duration=1800  # 30 menit per cycle
    )
    
    # Buat laporan
    print("\n📈 Generating comprehensive report...")
    generator.create_comprehensive_report(dataset)
    
    # Visualisasi
    print("\n🎨 Creating visualizations...")
    generator.visualize_dataset(dataset, 'driving_cycles_analysis.png')
    
    # Export dataset
    print("\n💾 Exporting dataset...")
    generator.export_dataset(dataset)
    
    # Rekomendasi akhir
    recommendations = generator.get_recommended_methods(dataset)
    
    print("\n" + "=" * 60)
    print("🎯 FINAL RECOMMENDATIONS")
    print("=" * 60)
    
    for domain, rec in recommendations.items():
        overall = rec['overall_recommendation']
        print(f"\n📍 {domain.upper():<10}: Use {overall['method']} method")
        print(f"   Average Similarity: {overall['average_similarity']:.3f}")
        print(f"   Description: {generator.domains[domain]['description']}")
    
    print("\n✅ Dataset generation completed successfully!")
    print(f"📁 Total cycles generated: {dataset['metadata']['total_cycles']}")
    
    return dataset, generator, recommendations

if __name__ == "__main__":
    dataset, generator, recommendations = main()