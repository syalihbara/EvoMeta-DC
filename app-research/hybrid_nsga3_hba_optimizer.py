# hybrid_nsga3_hba_optimizer.py
import numpy as np
import pandas as pd
import os
import sys
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import json
import time

# PyQt imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
                             QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox,
                             QProgressBar, QMessageBox, QFileDialog, QSplitter,
                             QHeaderView, QFormLayout, QLineEdit, QSizePolicy, QScrollArea,
                             QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

class HybridNSGA3_HBA:
    def __init__(self, population_size=50, max_generations=100, n_objectives=3, 
                 domains=['hilly', 'urban', 'suburban', 'congested', 'highway']):
        self.population_size = population_size
        self.max_generations = max_generations
        self.n_objectives = n_objectives
        self.domains = domains
        self.optimization_results = {}
        
        # NSGA-III parameters
        self.division_outer = 4
        self.division_inner = 0
        
        # HBA parameters
        self.C = 2.0  # exploration-exploitation balance
        self.beta = 6.0  # honey badger's ability to get food
        self.alpha = 1.0  # density factor
        
    def initialize_population(self, domain_data):
        """Initialize population based on domain characteristics"""
        population = []
        
        for domain in self.domains:
            domain_df = domain_data[domain_data['domain'] == domain]
            if len(domain_df) == 0:
                continue
                
            # Use existing cycles as initial population
            for _, cycle in domain_df.iterrows():
                individual = {
                    'cycle_id': cycle['cycle_id'],
                    'domain': domain,
                    'battery_Wh_per_km': cycle['battery_Wh_per_km'],
                    'regen_energy_Wh': cycle['regen_energy_Wh'],
                    'final_soc_percent': cycle['final_soc_percent'],
                    'c_rate_peak': cycle['c_rate_peak'],
                    'avg_speed': cycle['avg_speed'],
                    'distance_km': cycle['distance_km'],
                    'motor_efficiency_actual': cycle['motor_efficiency_actual']
                }
                population.append(individual)
                
        return population[:self.population_size]
    
    def calculate_objectives(self, individual, weights=None):
        """Calculate multiple objectives for optimization"""
        if weights is None:
            weights = [0.4, 0.3, 0.3]  # Default weights: efficiency, battery health, performance
            
        # Objective 1: Energy efficiency (minimize battery consumption)
        f1 = individual['battery_Wh_per_km']
        
        # Objective 2: Battery health (minimize peak C-rate, maximize final SOC)
        f2 = individual['c_rate_peak'] - (individual['final_soc_percent'] / 100)
        
        # Objective 3: Performance (balance between regen and efficiency)
        f3 = (individual['battery_Wh_per_km'] / individual['regen_energy_Wh']) if individual['regen_energy_Wh'] > 0 else float('inf')
        
        # Normalize objectives
        f1_norm = f1 / 300  # Normalize by typical max consumption
        f2_norm = f2 / 3.0  # Normalize by typical max C-rate
        f3_norm = min(f3 / 10.0, 1.0)  # Normalize performance metric
        
        objectives = np.array([f1_norm, f2_norm, f3_norm])
        weighted_score = np.dot(objectives, weights)
        
        return weighted_score, objectives
    
    def honey_badger_phase(self, population, best_solutions, domain):
        """Honey Badger Algorithm phase for exploration and exploitation"""
        new_population = []
        
        for i, individual in enumerate(population):
            if individual['domain'] != domain:
                new_population.append(individual)
                continue
                
            # Calculate intensity (fitness-based)
            fitness, _ = self.calculate_objectives(individual)
            intensity = 1 / (1 + fitness)
            
            # Density factor (decreases with iterations)
            alpha = self.alpha * (1 - (i / len(population)))
            
            # Exploration phase (digging mode)
            if np.random.random() < 0.5:
                # Move towards better solutions
                if best_solutions:
                    best_sol = np.random.choice(best_solutions)
                    new_individual = individual.copy()
                    
                    # Update parameters based on best solution
                    for key in ['battery_Wh_per_km', 'regen_energy_Wh', 'c_rate_peak']:
                        if key in new_individual and key in best_sol:
                            delta = best_sol[key] - new_individual[key]
                            new_individual[key] += alpha * intensity * delta * np.random.random()
                    
            # Exploitation phase (honey mode)
            else:
                # Local search around current solution
                new_individual = individual.copy()
                for key in ['battery_Wh_per_km', 'regen_energy_Wh', 'c_rate_peak']:
                    if key in new_individual:
                        perturbation = alpha * intensity * np.random.normal(0, 0.1)
                        new_individual[key] *= (1 + perturbation)
                        
                        # Ensure bounds
                        if key == 'battery_Wh_per_km':
                            new_individual[key] = max(50, min(400, new_individual[key]))
                        elif key == 'c_rate_peak':
                            new_individual[key] = max(0.5, min(5.0, new_individual[key]))
            
            new_population.append(new_individual)
            
        return new_population
    
    def non_dominated_sorting(self, population):
        """NSGA-III non-dominated sorting"""
        fronts = [[]]
        
        for i, ind1 in enumerate(population):
            ind1['dominated_by'] = 0
            ind1['dominates'] = []
            
            for j, ind2 in enumerate(population):
                if i != j:
                    dominates = self.dominates(ind1, ind2)
                    if dominates:
                        ind1['dominates'].append(j)
                    elif self.dominates(ind2, ind1):
                        ind1['dominated_by'] += 1
            
            if ind1['dominated_by'] == 0:
                fronts[0].append(i)
        
        i = 0
        while fronts[i]:
            next_front = []
            for ind_idx in fronts[i]:
                for dominated_idx in population[ind_idx]['dominates']:
                    population[dominated_idx]['dominated_by'] -= 1
                    if population[dominated_idx]['dominated_by'] == 0:
                        next_front.append(dominated_idx)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]
    
    def dominates(self, ind1, ind2):
        """Check if individual 1 dominates individual 2"""
        objectives1 = self.calculate_objectives(ind1)[1]
        objectives2 = self.calculate_objectives(ind2)[1]
        
        # All objectives are minimized
        better_in_all = all(obj1 <= obj2 for obj1, obj2 in zip(objectives1, objectives2))
        strictly_better = any(obj1 < obj2 for obj1, obj2 in zip(objectives1, objectives2))
        
        return better_in_all and strictly_better
    
    def optimize_domain(self, domain_data, domain, weight_scenarios=None):
        """Main optimization function for a specific domain"""
        if weight_scenarios is None:
            weight_scenarios = [
                [0.6, 0.2, 0.2],  # Efficiency-focused
                [0.2, 0.6, 0.2],  # Battery health-focused  
                [0.2, 0.2, 0.6],  # Performance-focused
                [0.4, 0.3, 0.3]   # Balanced
            ]
        
        domain_results = {}
        
        for scenario_idx, weights in enumerate(weight_scenarios):
            print(f"Optimizing {domain} with weights {weights}")
            
            # Initialize population for this domain
            population = self.initialize_population(domain_data)
            population = [ind for ind in population if ind['domain'] == domain]
            
            if not population:
                print(f"No data available for domain {domain}")
                continue
            
            best_solutions = []
            convergence_history = []
            
            for generation in range(self.max_generations):
                # Non-dominated sorting
                fronts = self.non_dominated_sorting(population)
                
                # Get best solutions from first front
                current_best = [population[idx] for idx in fronts[0] if idx < len(population)]
                best_solutions = current_best[:10]  # Keep top 10
                
                # Honey Badger phase
                population = self.honey_badger_phase(population, best_solutions, domain)
                
                # Calculate average fitness for convergence tracking
                avg_fitness = np.mean([self.calculate_objectives(ind, weights)[0] for ind in population])
                convergence_history.append(avg_fitness)
                
                if generation % 20 == 0:
                    print(f"Generation {generation}, Avg Fitness: {avg_fitness:.4f}")
            
            # Store results for this scenario
            scenario_name = f"scenario_{scenario_idx+1}"
            domain_results[scenario_name] = {
                'weights': weights,
                'best_solutions': best_solutions,
                'convergence': convergence_history,
                'optimal_solution': self.select_optimal_solution(best_solutions, weights)
            }
        
        return domain_results
    
    def select_optimal_solution(self, solutions, weights):
        """Select the single best solution from Pareto front"""
        if not solutions:
            return None
            
        best_score = float('inf')
        best_solution = None
        
        for solution in solutions:
            score, _ = self.calculate_objectives(solution, weights)
            if score < best_score:
                best_score = score
                best_solution = solution
                
        return best_solution
    
    def run_optimization(self, metadata_path):
        """Run optimization for all domains"""
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load metadata
        domain_data = pd.read_csv(metadata_path)
        
        for domain in self.domains:
            print(f"\n=== Optimizing {domain.upper()} domain ===")
            domain_results = self.optimize_domain(domain_data, domain)
            self.optimization_results[domain] = domain_results
            
        return self.optimization_results
    
    def save_optimal_indicators(self, output_dir):
        """Save optimal indicators for each domain to CSV file"""
        optimal_data = []
        
        for domain, scenarios in self.optimization_results.items():
            for scenario_name, scenario_data in scenarios.items():
                optimal_sol = scenario_data['optimal_solution']
                if optimal_sol:
                    optimal_data.append({
                        'domain': domain,
                        'scenario': scenario_name,
                        'weights': str(scenario_data['weights']),
                        'optimal_cycle_id': optimal_sol.get('cycle_id', 'N/A'),
                        'battery_Wh_per_km': optimal_sol.get('battery_Wh_per_km', 0),
                        'regen_energy_Wh': optimal_sol.get('regen_energy_Wh', 0),
                        'final_soc_percent': optimal_sol.get('final_soc_percent', 0),
                        'c_rate_peak': optimal_sol.get('c_rate_peak', 0),
                        'avg_speed': optimal_sol.get('avg_speed', 0),
                        'distance_km': optimal_sol.get('distance_km', 0),
                        'motor_efficiency_actual': optimal_sol.get('motor_efficiency_actual', 0),
                        'optimization_score': self.calculate_objectives(optimal_sol, scenario_data['weights'])[0]
                    })
        
        optimal_df = pd.DataFrame(optimal_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(output_dir, 'optimal_driving_cycle_indicators.csv')
        optimal_df.to_csv(output_path, index=False)
        
        # Save detailed results to JSON
        json_path = os.path.join(output_dir, 'optimization_detailed_results.json')
        with open(json_path, 'w') as f:
            # Convert to serializable format
            serializable_results = {}
            for domain, scenarios in self.optimization_results.items():
                serializable_results[domain] = {}
                for scenario, data in scenarios.items():
                    serializable_results[domain][scenario] = {
                        'weights': data['weights'],
                        'convergence': data['convergence'],
                        'optimal_solution': {k: (v if not isinstance(v, (np.float32, np.float64)) else float(v)) 
                                           for k, v in data['optimal_solution'].items()} 
                        if data['optimal_solution'] else None
                    }
            json.dump(serializable_results, f, indent=2)
        
        return output_path, optimal_df


class OptimizationThread(QThread):
    """Thread to run optimization without freezing GUI"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, optimizer, metadata_path):
        super().__init__()
        self.optimizer = optimizer
        self.metadata_path = metadata_path
        self.is_running = True

    def run(self):
        try:
            self.log.emit("Starting optimization process...")
            
            # Progress simulation
            for i in range(101):
                if not self.is_running:
                    return
                self.progress.emit(i)
                self.msleep(30)  # Reduced sleep time for better responsiveness
            
            # Run actual optimization
            self.log.emit("Loading metadata...")
            results = self.optimizer.run_optimization(self.metadata_path)
            
            if not self.is_running:
                return
                
            self.finished.emit(results)
            
        except Exception as e:
            if self.is_running:
                self.error.emit(str(e))

    def stop(self):
        self.is_running = False
        self.terminate()
        self.wait(2000)  # Wait max 2 seconds for thread to finish


class OptimizationGUI:
    """GUI for the hybrid optimization module"""
    def __init__(self, parent=None):
        self.parent = parent
        self.optimizer = HybridNSGA3_HBA()
        self.results = None
        self.optimization_thread = None
        
    def create_optimization_panel(self):
        """Create optimization control panel with proper scroll area"""
        # Create the main widget (not scroll area directly)
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(8)
        
        # Create scroll area for the entire content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create content widget for scroll area
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        
        # Main layout for content
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Hybrid NSGA-III + HBA Optimization")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #88c0d0; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Optimization parameters group
        param_group = QGroupBox("Optimization Parameters")
        param_layout = QFormLayout(param_group)
        param_layout.setLabelAlignment(Qt.AlignLeft)
        
        self.pop_size_spin = QSpinBox()
        self.pop_size_spin.setRange(20, 200)
        self.pop_size_spin.setValue(50)
        param_layout.addRow("Population Size:", self.pop_size_spin)
        
        self.max_gen_spin = QSpinBox()
        self.max_gen_spin.setRange(50, 500)
        self.max_gen_spin.setValue(100)
        param_layout.addRow("Max Generations:", self.max_gen_spin)
        
        layout.addWidget(param_group)
        
        # Weight scenarios group
        weight_group = QGroupBox("Optimization Scenarios")
        weight_layout = QVBoxLayout(weight_group)
        
        weight_info = QLabel(
            "Different weight combinations for multi-objective optimization:\n"
            "• Scenario 1: Efficiency-focused (60% efficiency, 20% battery health, 20% performance)\n"
            "• Scenario 2: Battery health-focused (20% efficiency, 60% battery health, 20% performance)\n"
            "• Scenario 3: Performance-focused (20% efficiency, 20% battery health, 60% performance)\n"
            "• Scenario 4: Balanced (40% efficiency, 30% battery health, 30% performance)"
        )
        weight_info.setStyleSheet("color: #d08770; font-style: italic; font-size: 10px;")
        weight_info.setWordWrap(True)
        weight_layout.addWidget(weight_info)
        
        layout.addWidget(weight_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run Optimization")
        self.run_btn.clicked.connect(self.run_optimization)
        button_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("Stop Optimization")
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
        # Progress bar
        self.opt_progress = QProgressBar()
        self.opt_progress.setVisible(False)
        layout.addWidget(self.opt_progress)
        
        # Results display
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Optimization results will appear here...")
        self.results_text.setStyleSheet("""
            font-family: 'Courier New', monospace; 
            font-size: 10px; 
            background-color: #2e3440; 
            color: #d8dee9;
            padding: 10px;
            border: 1px solid #4c566a;
            border-radius: 4px;
        """)
        self.results_text.setMinimumHeight(350)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        # Add stretch to push everything to the top
        layout.addStretch(1)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        return main_widget
    
    def run_optimization(self):
        """Run the optimization process"""
        try:
            # Update optimizer parameters
            self.optimizer.population_size = self.pop_size_spin.value()
            self.optimizer.max_generations = self.max_gen_spin.value()
            
            # Find metadata file
            metadata_path = "battery_analysis_dataset/metadata.csv"
            if not os.path.exists(metadata_path):
                QMessageBox.warning(self.parent, "Warning", 
                                  "Please generate or load a dataset first!\n\n"
                                  "Generate a dataset using the main interface before running optimization.")
                return
            
            # Reset UI state
            self.opt_progress.setVisible(True)
            self.opt_progress.setValue(0)
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.save_btn.setEnabled(False)
            self.results_text.clear()
            self.results_text.append("🚀 Starting Hybrid NSGA-III + HBA Optimization...")
            self.results_text.append("=" * 60)
            
            # Force UI update
            QApplication.processEvents()
            
            # Create and start optimization thread
            self.optimization_thread = OptimizationThread(self.optimizer, metadata_path)
            self.optimization_thread.progress.connect(self.opt_progress.setValue)
            self.optimization_thread.finished.connect(self.on_optimization_finished)
            self.optimization_thread.error.connect(self.on_optimization_error)
            self.optimization_thread.log.connect(self.on_optimization_log)
            self.optimization_thread.start()
            
        except Exception as e:
            self.on_optimization_error(str(e))
    
    def stop_optimization(self):
        """Stop the optimization process"""
        if self.optimization_thread and self.optimization_thread.isRunning():
            self.optimization_thread.stop()
            self.results_text.append("\n⏹️ Optimization stopped by user.")
            self.reset_ui_state()
    
    def reset_ui_state(self):
        """Reset UI to initial state"""
        self.opt_progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(self.results is not None)
    
    def on_optimization_log(self, message):
        """Handle optimization log messages"""
        self.results_text.append(f"📝 {message}")
        # Auto-scroll to bottom
        self.results_text.verticalScrollBar().setValue(
            self.results_text.verticalScrollBar().maximum()
        )
        QApplication.processEvents()
    
    def on_optimization_finished(self, results):
        """Handle optimization completion"""
        try:
            self.results = results
            self.opt_progress.setValue(100)
            
            # Add delay to ensure progress bar shows 100%
            time.sleep(0.5)
            
            self.reset_ui_state()
            self.display_results()
            
            self.results_text.append("\n✅ Optimization completed successfully!")
            
            # Show success message
            QMessageBox.information(self.parent, "Success", 
                                  "Optimization completed successfully!\n\n"
                                  "Optimal driving cycles have been found for all domains.")
            
        except Exception as e:
            self.on_optimization_error(str(e))
    
    def on_optimization_error(self, error_msg):
        """Handle optimization error"""
        self.results_text.append(f"\n❌ Error: {error_msg}")
        self.reset_ui_state()
        
        # Show error message
        QMessageBox.critical(self.parent, "Error", 
                           f"Optimization failed:\n{error_msg}")
    
    def display_results(self):
        """Display optimization results"""
        if not self.results:
            return
            
        results_text = "\n" + "=" * 60 + "\n"
        results_text += "🎯 OPTIMIZATION RESULTS\n"
        results_text += "=" * 60 + "\n\n"
        
        for domain, scenarios in self.results.items():
            results_text += f"🏔️  DOMAIN: {domain.upper()}\n"
            results_text += "─" * 50 + "\n"
            
            for scenario_name, scenario_data in scenarios.items():
                optimal_sol = scenario_data['optimal_solution']
                if optimal_sol:
                    results_text += f"\n📊 Scenario: {scenario_name}\n"
                    results_text += f"   Weights: {scenario_data['weights']}\n"
                    results_text += f"   🔗 Optimal Cycle: {optimal_sol.get('cycle_id', 'N/A')}\n"
                    results_text += f"   ⚡ Battery Consumption: {optimal_sol.get('battery_Wh_per_km', 0):.2f} Wh/km\n"
                    results_text += f"   🔄 Regen Energy: {optimal_sol.get('regen_energy_Wh', 0):.2f} Wh\n"
                    results_text += f"   🔋 Final SOC: {optimal_sol.get('final_soc_percent', 0):.2f}%\n"
                    results_text += f"   📈 Peak C-rate: {optimal_sol.get('c_rate_peak', 0):.2f}\n"
                    results_text += f"   🎯 Optimization Score: {self.optimizer.calculate_objectives(optimal_sol, scenario_data['weights'])[0]:.4f}\n"
                    results_text += "   " + "─" * 30 + "\n"
            
            results_text += "\n"
        
        self.results_text.append(results_text)
        
        # Ensure scroll bar is at the bottom
        self.results_text.verticalScrollBar().setValue(
            self.results_text.verticalScrollBar().maximum()
        )
    
    def save_results(self):
        """Save optimization results"""
        if not self.results:
            QMessageBox.warning(self.parent, "Warning", "No results to save!")
            return
            
        try:
            output_dir = "optimization_results"
            csv_path, optimal_df = self.optimizer.save_optimal_indicators(output_dir)
            
            # Update results text
            self.results_text.append(f"\n💾 Results saved to:\n{csv_path}")
            self.results_text.append(f"📄 Total optimal solutions: {len(optimal_df)}")
            
            QMessageBox.information(self.parent, "Success", 
                                  f"Results saved successfully!\n\n"
                                  f"📁 Directory: {output_dir}\n"
                                  f"📊 File: optimal_driving_cycle_indicators.csv\n"
                                  f"🔢 Total solutions: {len(optimal_df)}")
        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to save results:\n{str(e)}")


# Integration with main GUI
def add_optimization_tab(main_gui):
    """Add optimization tab to the main GUI"""
    optimizer_gui = OptimizationGUI(main_gui)
    optimization_panel = optimizer_gui.create_optimization_panel()
    
    # Add as new tab
    main_gui.tabs.addTab(optimization_panel, "🎯 NSGA3-HBA Optimization")
    
    return optimizer_gui


# Example usage for testing
if __name__ == "__main__":
    # Test the optimizer without GUI
    optimizer = HybridNSGA3_HBA(population_size=30, max_generations=50)
    
    # Create test data
    test_data = pd.DataFrame({
        'cycle_id': [f'test_{i}' for i in range(100)],
        'domain': np.random.choice(['urban', 'highway', 'hilly'], 100),
        'battery_Wh_per_km': np.random.uniform(100, 350, 100),
        'regen_energy_Wh': np.random.uniform(10, 300, 100),
        'final_soc_percent': np.random.uniform(20, 100, 100),
        'c_rate_peak': np.random.uniform(0.5, 3.0, 100),
        'avg_speed': np.random.uniform(10, 50, 100),
        'distance_km': np.random.uniform(1, 50, 100),
        'motor_efficiency_actual': np.random.uniform(0.7, 0.95, 100)
    })
    
    # Test optimization for one domain
    results = optimizer.optimize_domain(test_data, 'urban')
    print("Optimization completed!")
    
    # Save results
    output_path, optimal_df = optimizer.save_optimal_indicators('test_results')
    print(f"Results saved to: {output_path}")
    print(f"Optimal solutions:\n{optimal_df}")