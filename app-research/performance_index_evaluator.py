# performance_index_evaluator.py
import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# PyQt imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
                             QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox,
                             QProgressBar, QMessageBox, QFileDialog, QSplitter,
                             QHeaderView, QFormLayout, QLineEdit, QSizePolicy, QScrollArea,
                             QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

class PerformanceIndexEvaluator:
    def __init__(self):
        self.optimal_cycles = None
        self.performance_results = None
        
    def load_optimal_cycles(self, csv_path):
        """Load optimal cycles from CSV file"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Optimal cycles file not found: {csv_path}")
        
        self.optimal_cycles = pd.read_csv(csv_path)
        return self.optimal_cycles
    
    def calculate_eco_driving_index(self, cycle_data):
        """Calculate Eco Driving Index (EDI)"""
        # EDI components (all normalized to 0-1 where higher is better)
        components = {}
        
        # 1. Energy Efficiency (40% weight)
        battery_consumption = cycle_data['battery_Wh_per_km']
        efficiency_score = max(0, 1 - (battery_consumption / 400))  # Normalize to 0-1
        components['energy_efficiency'] = efficiency_score
        
        # 2. Regenerative Braking Utilization (25% weight)
        regen_energy = cycle_data['regen_energy_Wh']
        battery_energy = cycle_data['battery_energy_Wh']
        regen_ratio = regen_energy / battery_energy if battery_energy > 0 else 0
        regen_score = min(1.0, regen_ratio * 5)  # Normalize to 0-1
        components['regen_utilization'] = regen_score
        
        # 3. Speed Smoothness (20% weight)
        avg_speed = cycle_data['avg_speed']
        max_speed = cycle_data['max_speed']
        speed_variability = max_speed / avg_speed if avg_speed > 0 else 1
        smoothness_score = max(0, 1 - (speed_variability - 1) / 2)  # Normalize to 0-1
        components['speed_smoothness'] = smoothness_score
        
        # 4. Motor Efficiency (15% weight)
        motor_efficiency = cycle_data['motor_efficiency_actual']
        efficiency_score = (motor_efficiency - 0.7) / 0.25  # Normalize from 0.7-0.95 to 0-1
        components['motor_efficiency'] = max(0, min(1, efficiency_score))
        
        # Calculate weighted EDI
        weights = {
            'energy_efficiency': 0.40,
            'regen_utilization': 0.25,
            'speed_smoothness': 0.20,
            'motor_efficiency': 0.15
        }
        
        edi_score = sum(components[comp] * weights[comp] for comp in components)
        return min(1.0, max(0.0, edi_score)), components
    
    def calculate_battery_system_index(self, cycle_data):
        """Calculate Battery System Index (BSI)"""
        # BSI components (all normalized to 0-1 where higher is better)
        components = {}
        
        # 1. Battery Health (35% weight)
        peak_c_rate = cycle_data['c_rate_peak']
        c_rate_score = max(0, 1 - (peak_c_rate / 5))  # Normalize to 0-1
        components['c_rate_health'] = c_rate_score
        
        # 2. SOC Management (30% weight)
        final_soc = cycle_data['final_soc_percent']
        soc_consumed = cycle_data['soc_consumed_percent']
        
        # Prefer final SOC around 50% for battery longevity
        soc_deviation = abs(final_soc - 50) / 50
        soc_score = 1 - soc_deviation
        
        # Also consider SOC consumption rate
        soc_consumption_score = max(0, 1 - (soc_consumed / 30))
        
        soc_management_score = (soc_score + soc_consumption_score) / 2
        components['soc_management'] = soc_management_score
        
        # 3. Battery Stress (25% weight)
        stress_events = cycle_data.get('battery_stress_high_power_events', 0)
        stress_score = max(0, 1 - (stress_events / 10))  # Normalize to 0-1
        components['stress_management'] = stress_score
        
        # 4. Thermal Considerations (10% weight)
        # Simplified thermal score based on C-rate and efficiency
        thermal_score = (c_rate_score + cycle_data['motor_efficiency_actual']) / 2
        components['thermal_considerations'] = thermal_score
        
        # Calculate weighted BSI
        weights = {
            'c_rate_health': 0.35,
            'soc_management': 0.30,
            'stress_management': 0.25,
            'thermal_considerations': 0.10
        }
        
        bsi_score = sum(components[comp] * weights[comp] for comp in components)
        return min(1.0, max(0.0, bsi_score)), components
    
    def calculate_overall_performance_index(self, edi_score, bsi_score, weights=None):
        """Calculate Overall Performance Index (OPI)"""
        if weights is None:
            weights = [0.6, 0.4]  # Default: 60% EDI, 40% BSI
        
        opi_score = (edi_score * weights[0] + bsi_score * weights[1])
        return min(1.0, max(0.0, opi_score))
    
    def evaluate_all_cycles(self, weight_scenario='balanced'):
        """Evaluate all optimal cycles using performance indices"""
        if self.optimal_cycles is None:
            raise ValueError("No optimal cycles loaded. Please load optimal cycles first.")
        
        # Define weight scenarios
        weight_scenarios = {
            'eco_focused': [0.8, 0.2],    # 80% EDI, 20% BSI
            'battery_focused': [0.2, 0.8], # 20% EDI, 80% BSI
            'balanced': [0.5, 0.5],        # 50% EDI, 50% BSI
            'performance': [0.6, 0.4]      # 60% EDI, 40% BSI
        }
        
        weights = weight_scenarios.get(weight_scenario, [0.5, 0.5])
        
        results = []
        
        for _, cycle in self.optimal_cycles.iterrows():
            # Calculate indices
            edi_score, edi_components = self.calculate_eco_driving_index(cycle)
            bsi_score, bsi_components = self.calculate_battery_system_index(cycle)
            opi_score = self.calculate_overall_performance_index(edi_score, bsi_score, weights)
            
            result = {
                'cycle_id': cycle['cycle_id'],
                'domain': cycle['domain'],
                'scenario': cycle['scenario'],
                'edi_score': edi_score,
                'bsi_score': bsi_score,
                'opi_score': opi_score,
                'weights_used': weight_scenario,
                'battery_Wh_per_km': cycle['battery_Wh_per_km'],
                'regen_energy_Wh': cycle['regen_energy_Wh'],
                'final_soc_percent': cycle['final_soc_percent'],
                'c_rate_peak': cycle['c_rate_peak'],
                'avg_speed': cycle['avg_speed'],
                'motor_efficiency': cycle['motor_efficiency_actual']
            }
            
            # Add component scores
            result.update({f'edi_{k}': v for k, v in edi_components.items()})
            result.update({f'bsi_{k}': v for k, v in bsi_components.items()})
            
            results.append(result)
        
        self.performance_results = pd.DataFrame(results)
        
        # Rank cycles within each domain
        self.performance_results['domain_rank'] = self.performance_results.groupby('domain')['opi_score'].rank(ascending=False, method='dense')
        
        return self.performance_results
    
    def get_top_cycles_per_domain(self, top_n=3):
        """Get top N cycles for each domain based on OPI score"""
        if self.performance_results is None:
            raise ValueError("No performance results available. Please run evaluation first.")
        
        top_cycles = []
        
        for domain in self.performance_results['domain'].unique():
            domain_cycles = self.performance_results[self.performance_results['domain'] == domain]
            top_domain_cycles = domain_cycles.nlargest(top_n, 'opi_score')
            top_cycles.append(top_domain_cycles)
        
        return pd.concat(top_cycles, ignore_index=True)
    
    def get_best_overall_cycle(self):
        """Get the single best cycle across all domains"""
        if self.performance_results is None:
            raise ValueError("No performance results available. Please run evaluation first.")
        
        return self.performance_results.loc[self.performance_results['opi_score'].idxmax()]
    
    def save_performance_results(self, output_dir):
        """Save performance evaluation results"""
        if self.performance_results is None:
            raise ValueError("No performance results to save.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        csv_path = os.path.join(output_dir, 'performance_evaluation_results.csv')
        self.performance_results.to_csv(csv_path, index=False)
        
        # Save top cycles summary
        top_cycles = self.get_top_cycles_per_domain()
        top_cycles_path = os.path.join(output_dir, 'top_performing_cycles.csv')
        top_cycles.to_csv(top_cycles_path, index=False)
        
        # Save best overall cycle
        best_cycle = self.get_best_overall_cycle()
        best_cycle_path = os.path.join(output_dir, 'best_overall_cycle.csv')
        pd.DataFrame([best_cycle]).to_csv(best_cycle_path, index=False)
        
        return csv_path, top_cycles_path, best_cycle_path


class PerformanceIndexGUI:
    """GUI for Performance Index evaluation"""
    def __init__(self, parent=None):
        self.parent = parent
        self.evaluator = PerformanceIndexEvaluator()
        self.results = None
        
    def create_performance_panel(self):
        """Create performance index evaluation panel"""
        # Create main widget
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(8)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create content widget
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        
        # Main layout for content
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Performance Index Evaluation")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #88c0d0; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description
        description = QLabel(
            "Evaluate optimal driving cycles using Eco Driving Index (EDI) and Battery System Index (BSI)\n"
            "to identify the best performing cycles across different domains."
        )
        description.setStyleSheet("color: #d08770; font-style: italic; padding: 5px;")
        description.setAlignment(Qt.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Control group
        control_group = QGroupBox("Evaluation Parameters")
        control_layout = QVBoxLayout(control_group)
        
        # Weight scenario selection
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(QLabel("Weight Scenario:"))
        self.weight_combo = QComboBox()
        self.weight_combo.addItems([
            "Balanced (50% EDI, 50% BSI)",
            "Eco-focused (80% EDI, 20% BSI)", 
            "Battery-focused (20% EDI, 80% BSI)",
            "Performance (60% EDI, 40% BSI)"
        ])
        weight_layout.addWidget(self.weight_combo)
        weight_layout.addStretch()
        control_layout.addLayout(weight_layout)
        
        # Top N cycles
        topn_layout = QHBoxLayout()
        topn_layout.addWidget(QLabel("Top N cycles per domain:"))
        self.topn_spin = QSpinBox()
        self.topn_spin.setRange(1, 10)
        self.topn_spin.setValue(3)
        topn_layout.addWidget(self.topn_spin)
        topn_layout.addStretch()
        control_layout.addLayout(topn_layout)
        
        layout.addWidget(control_group)
        
        # Index explanation group
        explanation_group = QGroupBox("Performance Indices Explanation")
        explanation_layout = QVBoxLayout(explanation_group)
        
        explanation_text = QLabel(
            "📊 <b>Eco Driving Index (EDI):</b><br>"
            "• Energy Efficiency (40%): Battery consumption per km<br>"
            "• Regenerative Braking (25%): Effective use of regeneration<br>"
            "• Speed Smoothness (20%): Consistent speed profile<br>"
            "• Motor Efficiency (15%): Motor operational efficiency<br><br>"
            
            "🔋 <b>Battery System Index (BSI):</b><br>"
            "• Battery Health (35%): Peak C-rate impact<br>"
            "• SOC Management (30%): Optimal state of charge usage<br>"
            "• Stress Management (25%): High power events<br>"
            "• Thermal Considerations (10%): Thermal impact factors<br><br>"
            
            "⭐ <b>Overall Performance Index (OPI):</b><br>"
            "Combined score based on selected weight scenario"
        )
        explanation_text.setStyleSheet("color: #a3be8c; font-size: 10px; padding: 10px;")
        explanation_text.setWordWrap(True)
        explanation_layout.addWidget(explanation_text)
        
        layout.addWidget(explanation_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Optimal Cycles")
        self.load_btn.clicked.connect(self.load_optimal_cycles)
        button_layout.addWidget(self.load_btn)
        
        self.evaluate_btn = QPushButton("Evaluate Performance")
        self.evaluate_btn.clicked.connect(self.evaluate_performance)
        self.evaluate_btn.setEnabled(False)
        button_layout.addWidget(self.evaluate_btn)
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results display
        results_group = QGroupBox("Performance Evaluation Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Performance evaluation results will appear here...")
        self.results_text.setStyleSheet("""
            font-family: 'Courier New', monospace; 
            font-size: 9px; 
            background-color: #2e3440; 
            color: #d8dee9;
            padding: 10px;
            border: 1px solid #4c566a;
            border-radius: 4px;
        """)
        self.results_text.setMinimumHeight(400)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        # Add stretch
        layout.addStretch(1)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        return main_widget
    
    def load_optimal_cycles(self):
        """Load optimal cycles from file"""
        try:
            # Try to find the optimal cycles file
            csv_path = "optimization_results/optimal_driving_cycle_indicators.csv"
            
            if not os.path.exists(csv_path):
                # If not found, let user select file
                csv_path, _ = QFileDialog.getOpenFileName(
                    self.parent, 
                    "Select Optimal Cycles CSV File",
                    "",
                    "CSV Files (*.csv)"
                )
                
                if not csv_path:
                    return
            
            self.evaluator.load_optimal_cycles(csv_path)
            self.evaluate_btn.setEnabled(True)
            self.results_text.setText(f"✅ Loaded optimal cycles from:\n{csv_path}\n\n"
                                    f"📊 Total cycles: {len(self.evaluator.optimal_cycles)}")
            
            QMessageBox.information(self.parent, "Success", 
                                  f"Successfully loaded {len(self.evaluator.optimal_cycles)} optimal cycles!")
            
        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to load optimal cycles:\n{str(e)}")
    
    def evaluate_performance(self):
        """Evaluate performance indices"""
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Get selected weight scenario
            scenario_map = {
                0: 'balanced',
                1: 'eco_focused', 
                2: 'battery_focused',
                3: 'performance'
            }
            scenario = scenario_map.get(self.weight_combo.currentIndex(), 'balanced')
            
            # Update progress
            self.progress_bar.setValue(30)
            QApplication.processEvents()
            
            # Run evaluation
            self.results = self.evaluator.evaluate_all_cycles(scenario)
            
            # Update progress
            self.progress_bar.setValue(100)
            QApplication.processEvents()
            
            # Display results
            self.display_results()
            
            self.save_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            
            QMessageBox.information(self.parent, "Success", 
                                  "Performance evaluation completed successfully!")
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self.parent, "Error", f"Performance evaluation failed:\n{str(e)}")
    
    def display_results(self):
        """Display performance evaluation results"""
        if self.results is None:
            return
        
        results_text = "🎯 PERFORMANCE EVALUATION RESULTS\n"
        results_text += "=" * 70 + "\n\n"
        
        # Best overall cycle
        best_cycle = self.evaluator.get_best_overall_cycle()
        results_text += "🏆 BEST OVERALL CYCLE:\n"
        results_text += "─" * 40 + "\n"
        results_text += f"Cycle ID: {best_cycle['cycle_id']}\n"
        results_text += f"Domain: {best_cycle['domain']}\n"
        results_text += f"Scenario: {best_cycle['scenario']}\n"
        results_text += f"Overall Performance Index: {best_cycle['opi_score']:.4f}\n"
        results_text += f"Eco Driving Index: {best_cycle['edi_score']:.4f}\n"
        results_text += f"Battery System Index: {best_cycle['bsi_score']:.4f}\n\n"
        
        # Top cycles per domain
        top_n = self.topn_spin.value()
        top_cycles = self.evaluator.get_top_cycles_per_domain(top_n)
        
        results_text += f"📈 TOP {top_n} CYCLES PER DOMAIN:\n"
        results_text += "=" * 70 + "\n\n"
        
        for domain in self.results['domain'].unique():
            results_text += f"🏔️  DOMAIN: {domain.upper()}\n"
            results_text += "─" * 50 + "\n"
            
            domain_cycles = top_cycles[top_cycles['domain'] == domain]
            
            for _, cycle in domain_cycles.iterrows():
                results_text += f"\nRank #{int(cycle['domain_rank'])}:\n"
                results_text += f"  Cycle: {cycle['cycle_id']}\n"
                results_text += f"  Scenario: {cycle['scenario']}\n"
                results_text += f"  OPI: {cycle['opi_score']:.4f} | "
                results_text += f"EDI: {cycle['edi_score']:.4f} | "
                results_text += f"BSI: {cycle['bsi_score']:.4f}\n"
                results_text += f"  Battery: {cycle['battery_Wh_per_km']:.1f} Wh/km | "
                results_text += f"Regen: {cycle['regen_energy_Wh']:.1f} Wh\n"
                results_text += f"  SOC: {cycle['final_soc_percent']:.1f}% | "
                results_text += f"C-rate: {cycle['c_rate_peak']:.2f}\n"
                results_text += "  " + "─" * 35 + "\n"
            
            results_text += "\n"
        
        # Domain averages
        results_text += "📊 DOMAIN PERFORMANCE SUMMARY:\n"
        results_text += "=" * 70 + "\n\n"
        
        domain_summary = self.results.groupby('domain').agg({
            'opi_score': ['mean', 'max', 'min'],
            'edi_score': 'mean',
            'bsi_score': 'mean'
        }).round(4)
        
        for domain in domain_summary.index:
            results_text += f"{domain.upper()}:\n"
            results_text += f"  Avg OPI: {domain_summary.loc[domain, ('opi_score', 'mean')]} | "
            results_text += f"Max OPI: {domain_summary.loc[domain, ('opi_score', 'max')]} | "
            results_text += f"Min OPI: {domain_summary.loc[domain, ('opi_score', 'min')]}\n"
            results_text += f"  Avg EDI: {domain_summary.loc[domain, ('edi_score', 'mean')]} | "
            results_text += f"Avg BSI: {domain_summary.loc[domain, ('bsi_score', 'mean')]}\n\n"
        
        self.results_text.setText(results_text)
    
    def save_results(self):
        """Save performance evaluation results"""
        try:
            output_dir = "performance_evaluation_results"
            csv_path, top_cycles_path, best_cycle_path = self.evaluator.save_performance_results(output_dir)
            
            self.results_text.append(f"\n💾 Results saved to:\n")
            self.results_text.append(f"📄 Detailed results: {csv_path}\n")
            self.results_text.append(f"🏆 Top cycles: {top_cycles_path}\n")
            self.results_text.append(f"⭐ Best overall cycle: {best_cycle_path}\n")
            
            QMessageBox.information(self.parent, "Success",
                                  f"Performance results saved successfully!\n\n"
                                  f"📁 Directory: {output_dir}\n"
                                  f"📊 Files: 3 CSV files generated")
            
        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to save results:\n{str(e)}")


# Integration function
def add_performance_index_tab(main_gui):
    """Add performance index tab to the main GUI"""
    performance_gui = PerformanceIndexGUI(main_gui)
    performance_panel = performance_gui.create_performance_panel()
    
    # Add as new tab
    main_gui.tabs.addTab(performance_panel, "📈 Performance Index")
    
    return performance_gui


# Update the main GUI integration
def add_all_optimization_tabs(main_gui):
    """Add all optimization-related tabs to the main GUI"""
    # Add optimization tab
    optimizer_gui = add_optimization_tab(main_gui)
    
    # Add performance index tab
    performance_gui = add_performance_index_tab(main_gui)
    
    return optimizer_gui, performance_gui


# Example usage for testing
if __name__ == "__main__":
    # Test the performance evaluator
    evaluator = PerformanceIndexEvaluator()
    
    # Create test data
    test_data = pd.DataFrame({
        'cycle_id': [f'test_{i}' for i in range(20)],
        'domain': ['urban'] * 5 + ['highway'] * 5 + ['hilly'] * 5 + ['suburban'] * 5,
        'scenario': ['scenario_1'] * 10 + ['scenario_2'] * 10,
        'battery_Wh_per_km': np.random.uniform(100, 350, 20),
        'regen_energy_Wh': np.random.uniform(10, 300, 20),
        'final_soc_percent': np.random.uniform(20, 100, 20),
        'c_rate_peak': np.random.uniform(0.5, 3.0, 20),
        'avg_speed': np.random.uniform(10, 50, 20),
        'max_speed': np.random.uniform(40, 120, 20),
        'distance_km': np.random.uniform(1, 50, 20),
        'motor_efficiency_actual': np.random.uniform(0.7, 0.95, 20),
        'battery_energy_Wh': np.random.uniform(1000, 5000, 20),
        'soc_consumed_percent': np.random.uniform(1, 20, 20),
        'battery_stress_high_power_events': np.random.randint(0, 5, 20)
    })
    
    evaluator.optimal_cycles = test_data
    results = evaluator.evaluate_all_cycles()
    
    print("Performance evaluation completed!")
    print(f"Best overall cycle: {evaluator.get_best_overall_cycle()['cycle_id']}")
    
    top_cycles = evaluator.get_top_cycles_per_domain(2)
    print(f"Top cycles per domain:\n{top_cycles[['cycle_id', 'domain', 'opi_score']]}")