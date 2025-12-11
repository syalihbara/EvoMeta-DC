import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
import warnings
warnings.filterwarnings('ignore')

class UnifiedDrivingCycleGenerator:
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.domain_characteristics = {
            'highway': {'avg_speed': 85, 'idle_ratio': 0.05, 'max_speed': 120},
            'congested': {'avg_speed': 15, 'idle_ratio': 0.35, 'max_speed': 40},
            'urban': {'avg_speed': 35, 'idle_ratio': 0.20, 'max_speed': 70},
            'sub_urban': {'avg_speed': 50, 'idle_ratio': 0.12, 'max_speed': 85},
            'hilly': {'avg_speed': 35, 'idle_ratio': 0.15, 'max_speed': 65}
        }
        self.generated_cycles = {}
        self.validation_results = {}
        
    # ==================== 4 METODE ORIGINAL ====================
    
    def markov_chain_method(self, domain, duration_seconds=600, cycle_id=0):
        """Metode 1: Markov Chain"""
        chars = self.domain_characteristics[domain]
        
        # Transition matrix berdasarkan domain
        if domain == 'highway':
            transition_matrix = np.array([
                [0.05, 0.25, 0.20, 0.10, 0.15, 0.15, 0.10],
                [0.08, 0.20, 0.25, 0.12, 0.15, 0.12, 0.08],
                [0.05, 0.15, 0.20, 0.15, 0.20, 0.15, 0.10],
                [0.03, 0.10, 0.15, 0.20, 0.25, 0.15, 0.12],
                [0.10, 0.15, 0.12, 0.08, 0.25, 0.20, 0.10],
                [0.08, 0.12, 0.15, 0.10, 0.15, 0.30, 0.10],
                [0.05, 0.08, 0.12, 0.15, 0.15, 0.20, 0.25]
            ])
        elif domain == 'congested':
            transition_matrix = np.array([
                [0.30, 0.25, 0.15, 0.05, 0.10, 0.10, 0.05],
                [0.20, 0.25, 0.20, 0.05, 0.15, 0.10, 0.05],
                [0.15, 0.20, 0.20, 0.08, 0.18, 0.12, 0.07],
                [0.10, 0.15, 0.18, 0.12, 0.20, 0.15, 0.10],
                [0.25, 0.18, 0.12, 0.08, 0.20, 0.12, 0.05],
                [0.20, 0.15, 0.15, 0.10, 0.15, 0.20, 0.05],
                [0.15, 0.12, 0.15, 0.12, 0.15, 0.18, 0.13]
            ])
        else:  # urban, sub_urban, hilly
            transition_matrix = np.array([
                [0.15, 0.25, 0.20, 0.08, 0.12, 0.12, 0.08],
                [0.12, 0.20, 0.22, 0.10, 0.15, 0.13, 0.08],
                [0.08, 0.15, 0.20, 0.12, 0.18, 0.15, 0.12],
                [0.05, 0.10, 0.15, 0.15, 0.22, 0.18, 0.15],
                [0.15, 0.12, 0.10, 0.08, 0.25, 0.20, 0.10],
                [0.10, 0.15, 0.12, 0.08, 0.15, 0.30, 0.10],
                [0.08, 0.10, 0.12, 0.12, 0.15, 0.20, 0.23]
            ])
        
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        
        velocities = [0]
        current_state = 0
        
        for t in range(1, duration_seconds):
            current_velocity = velocities[-1]
            
            # Update velocity berdasarkan state
            if current_state == 0:    # idle
                new_velocity = 0
            elif current_state == 1:  # slow_accel
                new_velocity = current_velocity + np.random.uniform(0.3, 1.0)
            elif current_state == 2:  # medium_accel
                new_velocity = current_velocity + np.random.uniform(0.8, 2.0)
            elif current_state == 3:  # fast_accel
                new_velocity = current_velocity + np.random.uniform(1.5, 3.0)
            elif current_state == 4:  # decelerate
                new_velocity = current_velocity - np.random.uniform(1.0, 2.5)
            elif current_state == 5:  # cruise_slow
                variation = np.random.uniform(-0.8, 0.8)
                new_velocity = current_velocity + variation
            elif current_state == 6:  # cruise_fast
                variation = np.random.uniform(-1.5, 1.5)
                new_velocity = current_velocity + variation
            
            new_velocity = max(0, min(new_velocity, chars['max_speed']))
            velocities.append(new_velocity)
            
            # State transition
            current_state = np.random.choice(len(transition_matrix), p=transition_matrix[current_state])
        
        # Smoothing
        velocities = gaussian_filter1d(velocities, sigma=2)
        
        # Calculate acceleration
        acceleration = [0]
        for i in range(1, len(velocities)):
            accel = velocities[i] - velocities[i-1]
            acceleration.append(accel)
        
        cycle_data = {
            'time': np.arange(duration_seconds),
            'velocity': velocities,
            'acceleration': acceleration,
            'road_gradient': np.zeros(duration_seconds),
            'metadata': {
                'domain': domain,
                'method': 'Markov Chain',
                'cycle_id': cycle_id,
                'duration': duration_seconds
            }
        }
        
        return cycle_data
    
    def segment_based_method(self, domain, duration_seconds=600, cycle_id=0):
        """Metode 2: Segment-Based"""
        chars = self.domain_characteristics[domain]
        
        def generate_segment(segment_type, current_speed, duration):
            if segment_type == 'stop':
                return [0] * duration, 0
            elif segment_type == 'accel':
                target_speed = np.random.uniform(chars['avg_speed'] * 0.7, chars['avg_speed'] * 1.3)
                profile = []
                for i in range(duration):
                    progress = i / duration
                    acceleration = (target_speed - current_speed) * (1 - np.cos(progress * np.pi/2))
                    new_speed = current_speed + acceleration
                    profile.append(new_speed)
                return profile, profile[-1] if profile else current_speed
            elif segment_type == 'decel':
                target_speed = np.random.uniform(0, chars['avg_speed'] * 0.5)
                profile = []
                for i in range(duration):
                    progress = i / duration
                    deceleration = (current_speed - target_speed) * (1 - np.cos(progress * np.pi/2))
                    new_speed = current_speed - deceleration
                    profile.append(max(0, new_speed))
                return profile, profile[-1] if profile else current_speed
            elif segment_type == 'cruise':
                cruise_speed = np.random.uniform(chars['avg_speed'] * 0.8, chars['avg_speed'] * 1.2)
                profile = [cruise_speed + np.random.uniform(-2, 2) for _ in range(duration)]
                return profile, cruise_speed
        
        # Tentukan probabilitas segment berdasarkan domain
        if domain == 'highway':
            segment_probs = {'stop': 0.02, 'accel': 0.15, 'decel': 0.18, 'cruise': 0.65}
        elif domain == 'congested':
            segment_probs = {'stop': 0.35, 'accel': 0.25, 'decel': 0.25, 'cruise': 0.15}
        elif domain == 'urban':
            segment_probs = {'stop': 0.20, 'accel': 0.30, 'decel': 0.30, 'cruise': 0.20}
        elif domain == 'sub_urban':
            segment_probs = {'stop': 0.12, 'accel': 0.25, 'decel': 0.25, 'cruise': 0.38}
        else:  # hilly
            segment_probs = {'stop': 0.15, 'accel': 0.30, 'decel': 0.30, 'cruise': 0.25}
        
        velocities = [0]
        time = 1
        
        while time < duration_seconds:
            segment_type = np.random.choice(['stop', 'accel', 'decel', 'cruise'], 
                                          p=[segment_probs['stop'], segment_probs['accel'], 
                                             segment_probs['decel'], segment_probs['cruise']])
            
            segment_duration = min(np.random.randint(20, 120), duration_seconds - time)
            current_speed = velocities[-1]
            
            segment_profile, new_speed = generate_segment(segment_type, current_speed, segment_duration)
            velocities.extend(segment_profile)
            time += segment_duration
        
        velocities = velocities[:duration_seconds]
        if len(velocities) < duration_seconds:
            velocities.extend([0] * (duration_seconds - len(velocities)))
        
        # Smoothing
        velocities = gaussian_filter1d(velocities, sigma=2)
        velocities = np.clip(velocities, 0, chars['max_speed'])
        
        # Calculate acceleration
        acceleration = [0]
        for i in range(1, len(velocities)):
            accel = velocities[i] - velocities[i-1]
            acceleration.append(accel)
        
        cycle_data = {
            'time': np.arange(duration_seconds),
            'velocity': velocities.tolist(),
            'acceleration': acceleration,
            'road_gradient': np.zeros(duration_seconds),
            'metadata': {
                'domain': domain,
                'method': 'Segment Based',
                'cycle_id': cycle_id,
                'duration': duration_seconds
            }
        }
        
        return cycle_data
    
    def fourier_method(self, domain, duration_seconds=600, cycle_id=0):
        """Metode 3: Fourier Series"""
        chars = self.domain_characteristics[domain]
        
        t = np.linspace(0, duration_seconds/10, duration_seconds)
        
        # Base waveform
        velocity = np.zeros_like(t)
        
        # Frequency components berdasarkan domain
        if domain == 'highway':
            velocity += 15 * np.sin(2 * np.pi * t / 300)
            velocity += 8 * np.sin(2 * np.pi * t / 100)
            velocity += 3 * np.sin(2 * np.pi * t / 30)
            base_speed = chars['avg_speed']
        elif domain == 'congested':
            velocity += 8 * np.sin(2 * np.pi * t / 150)
            velocity += 5 * np.sin(2 * np.pi * t / 60)
            velocity += 3 * np.sin(2 * np.pi * t / 20)
            base_speed = chars['avg_speed'] * 0.8
        else:
            velocity += 12 * np.sin(2 * np.pi * t / 200)
            velocity += 6 * np.sin(2 * np.pi * t / 80)
            velocity += 3 * np.sin(2 * np.pi * t / 25)
            base_speed = chars['avg_speed']
        
        velocity += np.random.normal(0, 3, len(t))
        velocity += base_speed
        
        # Apply constraints
        velocity = np.maximum(0, velocity)
        velocity = np.minimum(chars['max_speed'], velocity)
        
        # Add stops based on domain
        stop_probability = chars['idle_ratio'] / 50
        i = 0
        while i < len(velocity):
            if np.random.random() < stop_probability and velocity[i] < 20:
                stop_duration = np.random.randint(10, 40)
                stop_end = min(i + stop_duration, len(velocity))
                velocity[i:stop_end] = 0
                i = stop_end
            else:
                i += 1
        
        # Smoothing
        velocity = gaussian_filter1d(velocity, sigma=2)
        
        # Calculate acceleration
        acceleration = np.zeros_like(velocity)
        acceleration[1:] = np.diff(velocity)
        
        cycle_data = {
            'time': list(range(duration_seconds)),
            'velocity': velocity.tolist(),
            'acceleration': acceleration.tolist(),
            'road_gradient': np.zeros(duration_seconds),
            'metadata': {
                'domain': domain,
                'method': 'Fourier Series',
                'cycle_id': cycle_id,
                'duration': duration_seconds
            }
        }
        
        return cycle_data
    
    def rule_based_fsm_method(self, domain, duration_seconds=600, cycle_id=0):
        """Metode 4: Rule-Based FSM - DIPERBAIKI"""
        chars = self.domain_characteristics[domain]
        
        class DrivingFSM:
            def __init__(self, domain, chars):
                self.domain = domain
                self.chars = chars
                self.states = self._setup_states()
                self.current_state = 'STOPPED'
                self.current_speed = 0
                self.state_timer = 0
                self.state_duration = np.random.randint(*self.states['STOPPED']['duration_range'])
                
            def _setup_states(self):
                """Setup states untuk semua domain dengan lengkap"""
                base_states = {
                    'STOPPED': {
                        'duration_range': (15, 45), 
                        'next_states': ['ACCEL_LIGHT', 'ACCEL_MEDIUM'],
                        'action': lambda: 0
                    },
                    'ACCEL_LIGHT': {
                        'duration_range': (20, 40), 
                        'next_states': ['CRUISE', 'DECEL'],
                        'action': lambda: min(self.current_speed + np.random.uniform(0.5, 1.5), self.chars['max_speed'])
                    },
                    'ACCEL_MEDIUM': {
                        'duration_range': (15, 30), 
                        'next_states': ['CRUISE', 'DECEL'],
                        'action': lambda: min(self.current_speed + np.random.uniform(1.0, 2.5), self.chars['max_speed'])
                    },
                    'CRUISE': {
                        'duration_range': (30, 90), 
                        'next_states': ['DECEL', 'ACCEL_LIGHT', 'STOPPED'],
                        'action': lambda: self.current_speed + np.random.uniform(-2, 2)
                    },
                    'DECEL': {
                        'duration_range': (15, 35), 
                        'next_states': ['STOPPED', 'CRUISE', 'ACCEL_LIGHT'],
                        'action': lambda: max(0, self.current_speed - np.random.uniform(1.0, 3.0))
                    }
                }
                
                # Domain-specific adjustments
                if self.domain == 'highway':
                    base_states['STOPPED']['duration_range'] = (5, 15)
                    base_states['CRUISE']['duration_range'] = (60, 180)
                    base_states['CRUISE']['action'] = lambda: self.current_speed + np.random.uniform(-3, 3)
                elif self.domain == 'congested':
                    base_states['STOPPED']['duration_range'] = (20, 60)
                    base_states['ACCEL_LIGHT']['duration_range'] = (10, 25)
                    base_states['ACCEL_LIGHT']['action'] = lambda: min(self.current_speed + np.random.uniform(0.3, 1.0), self.chars['max_speed'])
                    base_states['CRUISE']['duration_range'] = (10, 30)
                elif self.domain == 'urban':
                    base_states['STOPPED']['duration_range'] = (15, 45)
                    base_states['CRUISE']['duration_range'] = (20, 60)
                elif self.domain == 'sub_urban':
                    base_states['STOPPED']['duration_range'] = (10, 30)
                    base_states['CRUISE']['duration_range'] = (40, 120)
                elif self.domain == 'hilly':
                    base_states['ACCEL_LIGHT']['action'] = lambda: min(self.current_speed + np.random.uniform(0.8, 2.0), self.chars['max_speed'])
                    base_states['DECEL']['action'] = lambda: max(0, self.current_speed - np.random.uniform(1.5, 4.0))
                
                return base_states
                
            def execute_state(self):
                """Execute current state dan return speed"""
                state = self.states[self.current_state]
                self.current_speed = state['action']()
                self.current_speed = max(0, min(self.current_speed, self.chars['max_speed']))
                self.state_timer += 1
                return self.current_speed
                
            def should_transition(self):
                """Check jika harus transisi state"""
                return self.state_timer >= self.state_duration
                
            def transition_state(self):
                """Transisi ke state berikutnya"""
                current_state_info = self.states[self.current_state]
                next_state = np.random.choice(current_state_info['next_states'])
                self.current_state = next_state
                self.state_timer = 0
                self.state_duration = np.random.randint(*self.states[next_state]['duration_range'])
        
        # Initialize FSM
        fsm = DrivingFSM(domain, chars)
        velocities = []
        
        # Generate velocities
        for t in range(duration_seconds):
            speed = fsm.execute_state()
            velocities.append(speed)
            
            if fsm.should_transition():
                fsm.transition_state()
        
        # Smoothing
        velocities = gaussian_filter1d(velocities, sigma=2)
        velocities = np.clip(velocities, 0, chars['max_speed'])
        
        # Calculate acceleration
        acceleration = [0]
        for i in range(1, len(velocities)):
            accel = velocities[i] - velocities[i-1]
            acceleration.append(accel)
        
        cycle_data = {
            'time': np.arange(duration_seconds),
            'velocity': velocities.tolist(),
            'acceleration': acceleration,
            'road_gradient': np.zeros(duration_seconds),
            'metadata': {
                'domain': domain,
                'method': 'Rule Based FSM',
                'cycle_id': cycle_id,
                'duration': duration_seconds
            }
        }
        
        return cycle_data

    # ==================== NSGA-III OPTIMIZATION ====================
    
    class DrivingCycleOptimizationProblem(Problem):
        def __init__(self, domain, target_stats, duration=600):
            self.domain = domain
            self.target_stats = target_stats
            self.duration = duration
            
            # 8 parameters untuk dioptimasi
            xl = [0.3, target_stats['avg_speed']*0.5, 0.05, 0.5, 0.001, 1.0, 5, 0.3]
            xu = [0.7, target_stats['avg_speed']*1.5, 0.35, 2.5, 0.01, 5.0, 25, 0.7]
            
            super().__init__(n_var=8, n_obj=4, n_ieq_constr=2, xl=xl, xu=xu)
            
        def _evaluate(self, X, out, *args, **kwargs):
            n_individuals = X.shape[0]
            F = np.zeros((n_individuals, self.n_obj))
            G = np.zeros((n_individuals, self.n_ieq_constr))
            
            for i in range(n_individuals):
                try:
                    # Generate cycle dengan parameter
                    cycle_data = self._generate_optimized_cycle(X[i])
                    stats = self._calculate_statistics(cycle_data)
                    
                    # Objective functions (minimize)
                    F[i, 0] = self._objective_speed_match(stats)
                    F[i, 1] = self._objective_idle_match(stats)
                    F[i, 2] = self._objective_accel_realism(stats)
                    F[i, 3] = self._objective_smoothness(cycle_data)
                    
                    # Constraints
                    G[i, 0] = self._constraint_max_speed(stats)
                    G[i, 1] = self._constraint_accel_limits(stats)
                    
                except Exception as e:
                    F[i, :] = 10  # Penalty lebih kecil
                    G[i, :] = 10
            
            out["F"] = F
            out["G"] = G
        
        def _generate_optimized_cycle(self, params):
            """Generate cycle dengan parameter teroptimasi"""
            urban_ratio, avg_speed, idle_ratio, accel_intensity, stop_freq, smoothness, speed_var, cruise_ratio = params
            
            duration = self.duration
            t = np.linspace(0, duration/8, duration)
            
            # Base waveform dengan parameter teroptimasi
            velocity = np.zeros_like(t)
            
            # Multiple frequency components
            velocity += speed_var * np.sin(2 * np.pi * t / 200)
            velocity += (speed_var/2) * np.sin(2 * np.pi * t / 80)
            velocity += (speed_var/4) * np.sin(2 * np.pi * t / 25)
            
            velocity += np.random.normal(0, speed_var/3, len(t))
            velocity += avg_speed
            
            # Apply domain-specific constraints
            velocity = np.maximum(0, velocity)
            velocity = np.minimum(self.target_stats['max_speed'], velocity)
            
            # Add stops based on optimized frequency
            i = 0
            while i < len(velocity):
                if np.random.random() < stop_freq and velocity[i] < 25:
                    stop_duration = np.random.randint(10, 40)
                    stop_end = min(i + stop_duration, len(velocity))
                    velocity[i:stop_end] = 0
                    i = stop_end
                else:
                    i += 1
            
            # Apply smoothing
            velocity = gaussian_filter1d(velocity, sigma=smoothness)
            
            # Calculate acceleration
            acceleration = np.zeros_like(velocity)
            acceleration[1:] = np.diff(velocity)
            
            return {
                'velocity': velocity,
                'acceleration': acceleration,
                'time': np.arange(duration)
            }
        
        def _calculate_statistics(self, cycle_data):
            """Hitung statistik cycle"""
            velocity = cycle_data['velocity']
            acceleration = cycle_data['acceleration']
            
            stats = {}
            stats['avg_speed'] = np.mean(velocity)
            stats['max_speed'] = np.max(velocity)
            stats['min_speed'] = np.min(velocity)
            
            # Idle analysis
            idle_mask = velocity <= 1.0
            stats['idle_ratio'] = np.sum(idle_mask) / len(velocity)
            
            # Acceleration analysis
            accel_mask = acceleration > 0.5
            decel_mask = acceleration < -0.5
            stats['accel_ratio'] = np.sum(accel_mask) / len(acceleration)
            stats['decel_ratio'] = np.sum(decel_mask) / len(acceleration)
            
            significant_accel = acceleration[acceleration > 1.0]
            stats['avg_accel'] = np.mean(significant_accel) if len(significant_accel) > 0 else 0
            stats['max_accel'] = np.max(acceleration) if len(acceleration) > 0 else 0
            stats['max_decel'] = np.min(acceleration) if len(acceleration) > 0 else 0
            
            return stats
        
        # Objective functions
        def _objective_speed_match(self, stats):
            target = self.target_stats['avg_speed']
            return abs(stats['avg_speed'] - target) / target
        
        def _objective_idle_match(self, stats):
            target = self.target_stats['idle_ratio']
            return abs(stats['idle_ratio'] - target) / max(target, 0.01)
        
        def _objective_accel_realism(self, stats):
            accel_penalty = max(0, abs(stats['avg_accel']) - 2.0) / 2.0
            return accel_penalty
        
        def _objective_smoothness(self, cycle_data):
            acceleration = cycle_data['acceleration']
            jerk = np.diff(acceleration)
            return np.mean(np.abs(jerk)) if len(jerk) > 0 else 0
        
        # Constraints
        def _constraint_max_speed(self, stats):
            return stats['max_speed'] - self.target_stats['max_speed']
        
        def _constraint_accel_limits(self, stats):
            return max(abs(stats['max_accel']), abs(stats['max_decel'])) - 3.0
    
    def nsga3_optimized_method(self, domain, duration_seconds=600, cycle_id=0, population_size=30, n_generations=15):
        """Metode 5: NSGA-III Optimized - DIPERBAIKI dengan fallback"""
        print(f"🔬 Optimizing {domain} cycle with NSGA-III...")
        
        target_stats = self.domain_characteristics[domain]
        
        try:
            problem = self.DrivingCycleOptimizationProblem(domain, target_stats, duration_seconds)
            
            # Setup NSGA-III dengan parameter lebih konservatif
            ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=6)
            algorithm = NSGA3(
                pop_size=population_size,
                ref_dirs=ref_dirs,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.8, eta=12),
                mutation=PM(prob=0.15, eta=15),
                eliminate_duplicates=True
            )
            
            # Run optimization
            result = minimize(
                problem,
                algorithm,
                ('n_gen', n_generations),
                verbose=False,
                seed=42
            )
            
            if len(result.opt) == 0:
                raise ValueError("No solutions found")
                
            # Take the best solution
            best_solution = result.opt[0]
            cycle_data = problem._generate_optimized_cycle(best_solution.X)
            stats = problem._calculate_statistics(cycle_data)
            
            cycle_output = {
                'time': np.arange(duration_seconds),
                'velocity': cycle_data['velocity'].tolist(),
                'acceleration': cycle_data['acceleration'].tolist(),
                'road_gradient': np.zeros(duration_seconds),
                'metadata': {
                    'domain': domain,
                    'method': 'NSGA-III Optimized',
                    'cycle_id': cycle_id,
                    'duration': duration_seconds,
                    'optimization_score': np.sum(best_solution.F)
                }
            }
            
            print(f"✅ {domain} optimization completed. Score: {np.sum(best_solution.F):.4f}")
            return cycle_output
            
        except Exception as e:
            print(f"❌ NSGA-III failed for {domain}: {e}")
            print(f"🔄 Falling back to Fourier method for {domain}")
            # Fallback to Fourier method
            return self.fourier_method(domain, duration_seconds, cycle_id)

    # ==================== VALIDATION SYSTEM ====================
    
    def validate_cycle(self, cycle_data):
        """Sistem validasi komprehensif"""
        velocity = np.array(cycle_data['velocity'])
        acceleration = np.array(cycle_data['acceleration'])
        domain = cycle_data['metadata']['domain']
        method = cycle_data['metadata']['method']
        
        target_stats = self.domain_characteristics[domain]
        
        # Basic statistics
        stats = {}
        stats['avg_speed'] = np.mean(velocity)
        stats['max_speed'] = np.max(velocity)
        stats['min_speed'] = np.min(velocity)
        stats['speed_std'] = np.std(velocity)
        
        # Idle analysis
        idle_threshold = 1.0
        idle_mask = velocity <= idle_threshold
        stats['idle_ratio'] = np.sum(idle_mask) / len(velocity)
        
        # Acceleration analysis
        accel_threshold = 0.5
        decel_threshold = -0.5
        
        accel_mask = acceleration > accel_threshold
        decel_mask = acceleration < decel_threshold
        cruise_mask = ~(idle_mask | accel_mask | decel_mask)
        
        stats['accel_ratio'] = np.sum(accel_mask) / len(acceleration)
        stats['decel_ratio'] = np.sum(decel_mask) / len(acceleration)
        stats['cruise_ratio'] = np.sum(cruise_mask) / len(acceleration)
        
        # Significant accelerations
        significant_accel = acceleration[acceleration > 1.0]
        significant_decel = acceleration[acceleration < -1.0]
        
        stats['avg_accel'] = np.mean(significant_accel) if len(significant_accel) > 0 else 0
        stats['avg_decel'] = np.mean(significant_decel) if len(significant_decel) > 0 else 0
        stats['max_accel'] = np.max(acceleration) if len(acceleration) > 0 else 0
        stats['max_decel'] = np.min(acceleration) if len(acceleration) > 0 else 0
        
        # Realism scoring
        realism_scores = {}
        
        # Speed match (30%)
        speed_diff = abs(stats['avg_speed'] - target_stats['avg_speed'])
        speed_score = max(0, 1 - speed_diff / target_stats['avg_speed'])
        realism_scores['speed_match'] = speed_score
        
        # Idle ratio match (30%)
        idle_diff = abs(stats['idle_ratio'] - target_stats['idle_ratio'])
        idle_score = max(0, 1 - idle_diff / max(target_stats['idle_ratio'], 0.1))
        realism_scores['idle_match'] = idle_score
        
        # Acceleration realism (20%)
        accel_score = 1 - min(max(0, abs(stats['avg_accel']) - 1.5) / 1.5, 1)
        realism_scores['accel_realism'] = max(0, accel_score)
        
        # Speed variation (10%)
        variation_score = min(stats['speed_std'] / 20, 1.0)
        realism_scores['speed_variation'] = variation_score
        
        # Pattern distribution (10%)
        pattern_score = 1 - (abs(stats['accel_ratio'] - 0.25) + abs(stats['decel_ratio'] - 0.25)) / 0.5
        realism_scores['pattern_dist'] = max(0, pattern_score)
        
        # Overall score (weighted average)
        weights = {
            'speed_match': 0.3,
            'idle_match': 0.3,
            'accel_realism': 0.2,
            'speed_variation': 0.1,
            'pattern_dist': 0.1
        }
        
        overall_score = sum(realism_scores[metric] * weights[metric] for metric in realism_scores)
        
        # Pass/Fail criteria - lebih longgar
        criteria_met = (
            abs(stats['avg_speed'] - target_stats['avg_speed']) < target_stats['avg_speed'] * 0.4 and
            abs(stats['idle_ratio'] - target_stats['idle_ratio']) < 0.2 and
            stats['max_speed'] <= target_stats['max_speed'] * 1.1 and
            overall_score >= 0.5
        )
        
        validation_result = {
            'stats': stats,
            'realism_scores': realism_scores,
            'overall_score': overall_score,
            'criteria_met': criteria_met,
            'domain': domain,
            'method': method
        }
        
        return validation_result

    # ==================== DATASET GENERATION ====================
    
    def generate_comprehensive_dataset(self, domains, methods, cycles_per_domain, duration_range):
        """Generate dataset komprehensif dengan semua metode dan domain"""
        print("🚗 Generating Comprehensive Multi-Domain Driving Cycle Dataset")
        print("=" * 60)
        print(f"📊 Domains: {domains}")
        print(f"🔧 Methods: {methods}")
        print(f"📈 Cycles per domain: {cycles_per_domain}")
        
        dataset = []
        cycle_id = 0
        
        for domain in domains:
            print(f"\n🌍 DOMAIN: {domain.upper()}")
            print("-" * 30)
            
            for method in methods:
                print(f"   🔨 Method: {method}")
                
                for i in range(cycles_per_domain):
                    duration = np.random.randint(*duration_range)
                    
                    try:
                        if method == 'Markov Chain':
                            cycle_data = self.markov_chain_method(domain, duration, cycle_id)
                        elif method == 'Segment Based':
                            cycle_data = self.segment_based_method(domain, duration, cycle_id)
                        elif method == 'Fourier Series':
                            cycle_data = self.fourier_method(domain, duration, cycle_id)
                        elif method == 'Rule Based FSM':
                            cycle_data = self.rule_based_fsm_method(domain, duration, cycle_id)
                        elif method == 'NSGA-III Optimized':
                            cycle_data = self.nsga3_optimized_method(domain, duration, cycle_id)
                        
                        # Validate cycle
                        validation = self.validate_cycle(cycle_data)
                        cycle_data['metadata']['validation'] = validation
                        
                        dataset.append(cycle_data)
                        cycle_id += 1
                        
                        status = "✅ PASS" if validation['criteria_met'] else "❌ FAIL"
                        print(f"      Cycle {i+1}: {status} (Score: {validation['overall_score']:.3f})")
                        
                    except Exception as e:
                        print(f"      ❌ ERROR in {method} for {domain}: {str(e)[:50]}...")
                        # Skip this cycle and continue
                        continue
        
        print(f"\n✅ Dataset generation completed!")
        print(f"📁 Total cycles generated: {len(dataset)}")
        
        return dataset

    # ==================== ANALYSIS AND VISUALIZATION ====================
    
    def analyze_dataset(self, dataset):
        """Analisis komprehensif dataset"""
        analysis = {
            'method_performance': {},
            'domain_performance': {},
            'overall_stats': {}
        }
        
        # Initialize performance trackers
        methods = set()
        domains = set()
        
        for cycle in dataset:
            method = cycle['metadata']['method']
            domain = cycle['metadata']['domain']
            validation = cycle['metadata']['validation']
            
            methods.add(method)
            domains.add(domain)
            
            # Method performance
            if method not in analysis['method_performance']:
                analysis['method_performance'][method] = {
                    'scores': [], 'pass_count': 0, 'total_count': 0
                }
            
            analysis['method_performance'][method]['scores'].append(validation['overall_score'])
            analysis['method_performance'][method]['total_count'] += 1
            if validation['criteria_met']:
                analysis['method_performance'][method]['pass_count'] += 1
            
            # Domain performance
            if domain not in analysis['domain_performance']:
                analysis['domain_performance'][domain] = {
                    'scores': [], 'pass_count': 0, 'total_count': 0
                }
            
            analysis['domain_performance'][domain]['scores'].append(validation['overall_score'])
            analysis['domain_performance'][domain]['total_count'] += 1
            if validation['criteria_met']:
                analysis['domain_performance'][domain]['pass_count'] += 1
        
        # Calculate averages
        for method in analysis['method_performance']:
            scores = analysis['method_performance'][method]['scores']
            analysis['method_performance'][method]['avg_score'] = np.mean(scores)
            analysis['method_performance'][method]['pass_rate'] = \
                analysis['method_performance'][method]['pass_count'] / analysis['method_performance'][method]['total_count']
        
        for domain in analysis['domain_performance']:
            scores = analysis['domain_performance'][domain]['scores']
            analysis['domain_performance'][domain]['avg_score'] = np.mean(scores)
            analysis['domain_performance'][domain]['pass_rate'] = \
                analysis['domain_performance'][domain]['pass_count'] / analysis['domain_performance'][domain]['total_count']
        
        # Overall statistics
        all_scores = [cycle['metadata']['validation']['overall_score'] for cycle in dataset]
        analysis['overall_stats'] = {
            'avg_score': np.mean(all_scores),
            'std_score': np.std(all_scores),
            'pass_rate': sum(1 for cycle in dataset if cycle['metadata']['validation']['criteria_met']) / len(dataset),
            'total_cycles': len(dataset)
        }
        
        return analysis
    
    def visualize_comparison(self, dataset):
        """Visualisasi perbandingan semua metode"""
        analysis = self.analyze_dataset(dataset)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Method Performance Comparison
        methods = list(analysis['method_performance'].keys())
        avg_scores = [analysis['method_performance'][m]['avg_score'] for m in methods]
        pass_rates = [analysis['method_performance'][m]['pass_rate'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, avg_scores, width, label='Average Score', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, pass_rates, width, label='Pass Rate', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Methods')
        ax1.set_ylabel('Scores')
        ax1.set_title('Method Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Domain Performance
        domains = list(analysis['domain_performance'].keys())
        domain_scores = [analysis['domain_performance'][d]['avg_score'] for d in domains]
        domain_pass_rates = [analysis['domain_performance'][d]['pass_rate'] for d in domains]
        
        x_domain = np.arange(len(domains))
        
        bars3 = ax2.bar(x_domain - width/2, domain_scores, width, label='Avg Score', alpha=0.8, color='lightgreen')
        bars4 = ax2.bar(x_domain + width/2, domain_pass_rates, width, label='Pass Rate', alpha=0.8, color='orange')
        
        ax2.set_xlabel('Domains')
        ax2.set_ylabel('Scores')
        ax2.set_title('Domain Performance Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_domain)
        ax2.set_xticklabels(domains, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Score Distribution by Method
        method_scores = {}
        for cycle in dataset:
            method = cycle['metadata']['method']
            score = cycle['metadata']['validation']['overall_score']
            if method not in method_scores:
                method_scores[method] = []
            method_scores[method].append(score)
        
        if method_scores:
            box_data = [method_scores[method] for method in methods if method in method_scores and method_scores[method]]
            ax3.boxplot(box_data, labels=[method for method in methods if method in method_scores and method_scores[method]])
            ax3.set_xticklabels([method for method in methods if method in method_scores and method_scores[method]], rotation=45)
            ax3.set_ylabel('Validation Score')
            ax3.set_title('Score Distribution by Method', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Best Cycles from Each Method
        best_cycles = {}
        for cycle in dataset:
            method = cycle['metadata']['method']
            score = cycle['metadata']['validation']['overall_score']
            if method not in best_cycles or score > best_cycles[method]['score']:
                best_cycles[method] = {'cycle': cycle, 'score': score}
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        for i, method in enumerate(methods):
            if method in best_cycles:
                cycle_data = best_cycles[method]['cycle']
                velocity = cycle_data['velocity'][:300]  # First 300 seconds
                time = cycle_data['time'][:300]
                ax4.plot(time, velocity, color=colors[i], label=f'{method} ({best_cycles[method]["score"]:.3f})', linewidth=1.5)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity (km/h)')
        ax4.set_title('Best Cycles from Each Method (First 300s)', fontsize=14, fontweight='bold')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return analysis
    
    def export_dataset(self, dataset, filename="unified_driving_cycles.csv"):
        """Export dataset ke CSV"""
        all_data = []
        
        for cycle in dataset:
            cycle_df = pd.DataFrame({
                'time': cycle['time'],
                'velocity': cycle['velocity'],
                'acceleration': cycle['acceleration'],
                'road_gradient': cycle['road_gradient'],
                'domain': cycle['metadata']['domain'],
                'method': cycle['metadata']['method'],
                'cycle_id': cycle['metadata']['cycle_id'],
                'duration': cycle['metadata']['duration'],
                'validation_score': cycle['metadata']['validation']['overall_score'],
                'validation_pass': cycle['metadata']['validation']['criteria_met']
            })
            
            # Add statistics
            stats = cycle['metadata']['validation']['stats']
            for key, value in stats.items():
                cycle_df[f'stat_{key}'] = value
            
            all_data.append(cycle_df)
        
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(filename, index=False)
        
        print(f"✅ Dataset exported to {filename}")
        print(f"📊 Total records: {len(final_df)}")
        
        return final_df

# ==================== DEMONSTRATION ====================
def demonstrate_unified_generator():
    """Demonstrasi lengkap generator terunifikasi"""
    
    print("🚗 UNIFIED DRIVING CYCLE GENERATOR")
    print("🔧 Combining 4 Methods + NSGA-III Optimization")
    print("=" * 60)
    
    # Initialize generator
    generator = UnifiedDrivingCycleGenerator(random_seed=42)
    
    # Generate comprehensive dataset
    dataset = generator.generate_comprehensive_dataset(
        domains=['highway', 'urban', 'congested'],
        methods=['Markov Chain', 'Segment Based', 'Fourier Series', 'Rule Based FSM', 'NSGA-III Optimized'],
        cycles_per_domain=2,
        duration_range=(300, 500)
    )
    
    if len(dataset) == 0:
        print("❌ No cycles generated. Please check the implementation.")
        return generator, [], {}
    
    # Analyze dataset
    print("\n📊 ANALYZING DATASET...")
    analysis = generator.analyze_dataset(dataset)
    
    # Print results
    print("\n" + "="*50)
    print("📈 OVERALL RESULTS")
    print("="*50)
    print(f"Total Cycles: {analysis['overall_stats']['total_cycles']}")
    print(f"Average Score: {analysis['overall_stats']['avg_score']:.3f}")
    print(f"Overall Pass Rate: {analysis['overall_stats']['pass_rate']:.1%}")
    
    print("\n🏆 METHOD PERFORMANCE:")
    print("-" * 30)
    for method, perf in analysis['method_performance'].items():
        print(f"{method:20} | Score: {perf['avg_score']:.3f} | Pass Rate: {perf['pass_rate']:.1%}")
    
    print("\n🌍 DOMAIN PERFORMANCE:")
    print("-" * 30)
    for domain, perf in analysis['domain_performance'].items():
        print(f"{domain:15} | Score: {perf['avg_score']:.3f} | Pass Rate: {perf['pass_rate']:.1%}")
    
    # Visualize comparison
    print("\n📈 GENERATING VISUALIZATIONS...")
    generator.visualize_comparison(dataset)
    
    # Export dataset
    print("\n💾 EXPORTING DATASET...")
    df = generator.export_dataset(dataset, "unified_multi_domain_cycles.csv")
    
    # Show best method
    if analysis['method_performance']:
        best_method = max(analysis['method_performance'].items(), key=lambda x: x[1]['avg_score'])
        print(f"\n🎯 RECOMMENDATION: Best method is '{best_method[0]}' with score {best_method[1]['avg_score']:.3f}")
    
    return generator, dataset, analysis

if __name__ == "__main__":
    generator, dataset, analysis = demonstrate_unified_generator()