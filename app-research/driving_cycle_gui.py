import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox, 
                             QTextEdit, QTabWidget, QGroupBox, QCheckBox, QProgressBar,
                             QFileDialog, QMessageBox, QSplitter, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QSlider, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

# Import generator class (pastikan file generator ada di direktori yang sama)
try:
    from driving_cycle_generator import UnifiedDrivingCycleGenerator
except ImportError:
    # Fallback jika tidak ada generator
    class UnifiedDrivingCycleGenerator:
        def __init__(self):
            pass

class GenerationThread(QThread):
    """Thread untuk generate dataset agar tidak freeze GUI"""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    
    def __init__(self, generator, domains, methods, cycles_per_domain, duration_range):
        super().__init__()
        self.generator = generator
        self.domains = domains
        self.methods = methods
        self.cycles_per_domain = cycles_per_domain
        self.duration_range = duration_range
    
    def run(self):
        try:
            total_cycles = len(self.domains) * len(self.methods) * self.cycles_per_domain
            current_progress = 0
            
            dataset = []
            cycle_id = 0
            
            for domain_idx, domain in enumerate(self.domains):
                self.progress_signal.emit(
                    int((domain_idx / len(self.domains)) * 100),
                    f"Processing domain: {domain}"
                )
                
                for method_idx, method in enumerate(self.methods):
                    for cycle_idx in range(self.cycles_per_domain):
                        duration = np.random.randint(*self.duration_range)
                        
                        try:
                            if method == 'Markov Chain':
                                cycle_data = self.generator.markov_chain_method(domain, duration, cycle_id)
                            elif method == 'Segment Based':
                                cycle_data = self.generator.segment_based_method(domain, duration, cycle_id)
                            elif method == 'Fourier Series':
                                cycle_data = self.generator.fourier_method(domain, duration, cycle_id)
                            elif method == 'Rule Based FSM':
                                cycle_data = self.generator.rule_based_fsm_method(domain, duration, cycle_id)
                            elif method == 'NSGA-III Optimized':
                                cycle_data = self.generator.nsga3_optimized_method(domain, duration, cycle_id)
                            
                            # Validate cycle
                            validation = self.generator.validate_cycle(cycle_data)
                            cycle_data['metadata']['validation'] = validation
                            dataset.append(cycle_data)
                            cycle_id += 1
                            
                        except Exception as e:
                            print(f"Error in {method} for {domain}: {e}")
                        
                        current_progress += 1
                        progress_percent = int((current_progress / total_cycles) * 100)
                        self.progress_signal.emit(
                            progress_percent,
                            f"Generated {current_progress}/{total_cycles} cycles"
                        )
            
            self.finished_signal.emit(dataset)
            
        except Exception as e:
            self.error_signal.emit(str(e))

class MatplotlibCanvas(FigureCanvas):
    """Canvas untuk plot matplotlib"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        self.fig.tight_layout()

class DrivingCycleGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.generator = UnifiedDrivingCycleGenerator()
        self.current_dataset = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("EV Driving Cycle Generator - Multi Domain")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Electric Vehicle Driving Cycle Generator")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("padding: 10px; background-color: #2c3e50; color: white;")
        main_layout.addWidget(title)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Setup tabs
        self.setup_generation_tab()
        self.setup_visualization_tab()
        self.setup_analysis_tab()
        self.setup_export_tab()
        
        # Status bar
        self.statusBar().showMessage("Ready to generate driving cycles")
        
    def setup_generation_tab(self):
        """Tab untuk generate dataset"""
        generation_tab = QWidget()
        layout = QVBoxLayout(generation_tab)
        
        # Configuration group
        config_group = QGroupBox("Generation Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # Domain selection
        domain_layout = QHBoxLayout()
        domain_layout.addWidget(QLabel("Domains:"))
        self.domain_checkboxes = {}
        domains = ['highway', 'congested', 'urban', 'sub_urban', 'hilly']
        for domain in domains:
            cb = QCheckBox(domain.title())
            cb.setChecked(True)
            self.domain_checkboxes[domain] = cb
            domain_layout.addWidget(cb)
        domain_layout.addStretch()
        config_layout.addLayout(domain_layout)
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Methods:"))
        self.method_checkboxes = {}
        methods = ['Markov Chain', 'Segment Based', 'Fourier Series', 'Rule Based FSM', 'NSGA-III Optimized']
        for method in methods:
            cb = QCheckBox(method)
            cb.setChecked(True)
            self.method_checkboxes[method] = cb
            method_layout.addWidget(cb)
        method_layout.addStretch()
        config_layout.addLayout(method_layout)
        
        # Parameters
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Cycles per Domain:"))
        self.cycles_spin = QSpinBox()
        self.cycles_spin.setRange(1, 20)
        self.cycles_spin.setValue(3)
        param_layout.addWidget(self.cycles_spin)
        
        param_layout.addWidget(QLabel("Min Duration (s):"))
        self.min_duration_spin = QSpinBox()
        self.min_duration_spin.setRange(60, 1800)
        self.min_duration_spin.setValue(300)
        param_layout.addWidget(self.min_duration_spin)
        
        param_layout.addWidget(QLabel("Max Duration (s):"))
        self.max_duration_spin = QSpinBox()
        self.max_duration_spin.setRange(300, 3600)
        self.max_duration_spin.setValue(600)
        param_layout.addWidget(self.max_duration_spin)
        
        param_layout.addStretch()
        config_layout.addLayout(param_layout)
        
        layout.addWidget(config_group)
        
        # Progress section
        progress_group = QGroupBox("Generation Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready to generate")
        progress_layout.addWidget(self.progress_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Dataset")
        self.generate_btn.clicked.connect(self.generate_dataset)
        self.generate_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-weight: bold; }")
        button_layout.addWidget(self.generate_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_generation)
        self.cancel_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_btn)
        
        progress_layout.addLayout(button_layout)
        layout.addWidget(progress_group)
        
        # Results summary
        results_group = QGroupBox("Generation Results")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)
        layout.addWidget(results_group)
        
        self.tabs.addTab(generation_tab, "Generation")
        
    def setup_visualization_tab(self):
        """Tab untuk visualisasi dataset"""
        visualization_tab = QWidget()
        layout = QVBoxLayout(visualization_tab)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("Domain:"))
        self.viz_domain_combo = QComboBox()
        self.viz_domain_combo.addItems(['All', 'highway', 'congested', 'urban', 'sub_urban', 'hilly'])
        control_layout.addWidget(self.viz_domain_combo)
        
        control_layout.addWidget(QLabel("Method:"))
        self.viz_method_combo = QComboBox()
        self.viz_method_combo.addItems(['All', 'Markov Chain', 'Segment Based', 'Fourier Series', 'Rule Based FSM', 'NSGA-III Optimized'])
        control_layout.addWidget(self.viz_method_combo)
        
        control_layout.addWidget(QLabel("Cycle ID:"))
        self.viz_cycle_spin = QSpinBox()
        self.viz_cycle_spin.setRange(0, 1000)
        control_layout.addWidget(self.viz_cycle_spin)
        
        self.viz_btn = QPushButton("Update Visualization")
        self.viz_btn.clicked.connect(self.update_visualization)
        control_layout.addWidget(self.viz_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Matplotlib canvas
        self.viz_canvas = MatplotlibCanvas(self, width=10, height=8)
        layout.addWidget(self.viz_canvas)
        
        self.tabs.addTab(visualization_tab, "Visualization")
        
    def setup_analysis_tab(self):
        """Tab untuk analisis dataset"""
        analysis_tab = QWidget()
        layout = QVBoxLayout(analysis_tab)
        
        # Analysis controls
        analysis_control_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze Dataset")
        self.analyze_btn.clicked.connect(self.analyze_dataset)
        analysis_control_layout.addWidget(self.analyze_btn)
        analysis_control_layout.addStretch()
        layout.addLayout(analysis_control_layout)
        
        # Analysis results
        self.analysis_text = QTextEdit()
        layout.addWidget(self.analysis_text)
        
        self.tabs.addTab(analysis_tab, "Analysis")
        
    def setup_export_tab(self):
        """Tab untuk export dataset"""
        export_tab = QWidget()
        layout = QVBoxLayout(export_tab)
        
        # Export controls
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)
        
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Export Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["CSV", "Excel", "JSON"])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        export_layout.addLayout(format_layout)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path_edit = QTextEdit()
        self.file_path_edit.setMaximumHeight(30)
        self.file_path_edit.setText("driving_cycles.csv")
        file_layout.addWidget(QLabel("File:"))
        file_layout.addWidget(self.file_path_edit)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_btn)
        export_layout.addLayout(file_layout)
        
        layout.addWidget(export_group)
        
        # Export button
        self.export_btn = QPushButton("Export Dataset")
        self.export_btn.clicked.connect(self.export_dataset)
        self.export_btn.setStyleSheet("QPushButton { background-color: #2980b9; color: white; font-weight: bold; }")
        layout.addWidget(self.export_btn)
        
        # Export status
        self.export_status = QTextEdit()
        self.export_status.setMaximumHeight(100)
        layout.addWidget(self.export_status)
        
        self.tabs.addTab(export_tab, "Export")
        
    def generate_dataset(self):
        """Generate dataset berdasarkan konfigurasi"""
        # Get selected domains
        selected_domains = [domain for domain, cb in self.domain_checkboxes.items() if cb.isChecked()]
        if not selected_domains:
            QMessageBox.warning(self, "Warning", "Please select at least one domain")
            return
        
        # Get selected methods
        selected_methods = [method for method, cb in self.method_checkboxes.items() if cb.isChecked()]
        if not selected_methods:
            QMessageBox.warning(self, "Warning", "Please select at least one method")
            return
        
        # Setup UI for generation
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting generation...")
        
        # Start generation thread
        duration_range = (self.min_duration_spin.value(), self.max_duration_spin.value())
        self.generation_thread = GenerationThread(
            self.generator, selected_domains, selected_methods, 
            self.cycles_spin.value(), duration_range
        )
        self.generation_thread.progress_signal.connect(self.update_progress)
        self.generation_thread.finished_signal.connect(self.generation_finished)
        self.generation_thread.error_signal.connect(self.generation_error)
        self.generation_thread.start()
        
    def cancel_generation(self):
        """Cancel generation process"""
        if hasattr(self, 'generation_thread') and self.generation_thread.isRunning():
            self.generation_thread.terminate()
            self.generation_thread.wait()
        self.reset_generation_ui()
        self.statusBar().showMessage("Generation cancelled")
        
    def reset_generation_ui(self):
        """Reset UI setelah generation selesai/dibatalkan"""
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
    def update_progress(self, value, message):
        """Update progress bar dan label"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        
    def generation_finished(self, dataset):
        """Handle ketika generation selesai"""
        self.current_dataset = dataset
        self.reset_generation_ui()
        
        # Update results summary
        total_cycles = len(dataset)
        domains = set(cycle['metadata']['domain'] for cycle in dataset)
        methods = set(cycle['metadata']['method'] for cycle in dataset)
        
        # Calculate average score
        avg_score = np.mean([cycle['metadata']['validation']['overall_score'] for cycle in dataset])
        pass_count = sum(1 for cycle in dataset if cycle['metadata']['validation']['criteria_met'])
        
        summary = f"""Generation Completed!
Total Cycles: {total_cycles}
Domains: {', '.join(domains)}
Methods: {', '.join(methods)}
Average Validation Score: {avg_score:.3f}
Passed Validation: {pass_count}/{total_cycles} ({pass_count/total_cycles*100:.1f}%)
"""
        self.results_text.setText(summary)
        self.statusBar().showMessage(f"Generation completed: {total_cycles} cycles generated")
        
        # Enable analysis and export tabs
        self.analyze_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
    def generation_error(self, error_message):
        """Handle error selama generation"""
        self.reset_generation_ui()
        QMessageBox.critical(self, "Generation Error", f"Error during generation:\n{error_message}")
        self.statusBar().showMessage("Generation failed")
        
    def update_visualization(self):
        """Update visualization berdasarkan selection"""
        if self.current_dataset is None:
            QMessageBox.information(self, "Info", "No dataset available. Please generate dataset first.")
            return
            
        self.viz_canvas.axes.clear()
        
        selected_domain = self.viz_domain_combo.currentText()
        selected_method = self.viz_method_combo.currentText()
        selected_cycle_id = self.viz_cycle_spin.value()
        
        # Filter cycles
        filtered_cycles = self.current_dataset
        if selected_domain != 'All':
            filtered_cycles = [c for c in filtered_cycles if c['metadata']['domain'] == selected_domain]
        if selected_method != 'All':
            filtered_cycles = [c for c in filtered_cycles if c['metadata']['method'] == selected_method]
        if selected_cycle_id > 0:
            filtered_cycles = [c for c in filtered_cycles if c['metadata']['cycle_id'] == selected_cycle_id]
            
        if not filtered_cycles:
            self.viz_canvas.axes.text(0.5, 0.5, 'No cycles match the criteria', 
                                    ha='center', va='center', transform=self.viz_canvas.axes.transAxes)
            self.viz_canvas.draw()
            return
            
        # Plot first matching cycle
        cycle = filtered_cycles[0]
        time = cycle['time']
        velocity = cycle['velocity']
        
        # Create plot
        self.viz_canvas.axes.plot(time, velocity, 'b-', linewidth=1, alpha=0.8)
        self.viz_canvas.axes.set_xlabel('Time (s)')
        self.viz_canvas.axes.set_ylabel('Velocity (km/h)')
        
        metadata = cycle['metadata']
        validation = metadata['validation']
        title = f"{metadata['domain'].title()} - {metadata['method']} (Cycle {metadata['cycle_id']})"
        title += f"\nScore: {validation['overall_score']:.3f} - {'PASS' if validation['criteria_met'] else 'FAIL'}"
        self.viz_canvas.axes.set_title(title)
        self.viz_canvas.axes.grid(True, alpha=0.3)
        
        self.viz_canvas.draw()
        
    def analyze_dataset(self):
        """Analyze current dataset"""
        if self.current_dataset is None:
            QMessageBox.information(self, "Info", "No dataset available. Please generate dataset first.")
            return
            
        try:
            analysis = self.generator.analyze_dataset(self.current_dataset)
            
            # Format analysis results
            analysis_text = "=== DATASET ANALYSIS ===\n\n"
            
            # Overall stats
            overall = analysis['overall_stats']
            analysis_text += f"OVERALL STATISTICS:\n"
            analysis_text += f"Total Cycles: {overall['total_cycles']}\n"
            analysis_text += f"Average Score: {overall['avg_score']:.3f}\n"
            analysis_text += f"Standard Deviation: {overall['std_score']:.3f}\n"
            analysis_text += f"Pass Rate: {overall['pass_rate']:.1%}\n\n"
            
            # Method performance
            analysis_text += "METHOD PERFORMANCE:\n"
            for method, perf in analysis['method_performance'].items():
                analysis_text += f"{method:20} | Score: {perf['avg_score']:.3f} | Pass Rate: {perf['pass_rate']:.1%}\n"
            analysis_text += "\n"
            
            # Domain performance
            analysis_text += "DOMAIN PERFORMANCE:\n"
            for domain, perf in analysis['domain_performance'].items():
                analysis_text += f"{domain:15} | Score: {perf['avg_score']:.3f} | Pass Rate: {perf['pass_rate']:.1%}\n"
            
            self.analysis_text.setText(analysis_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Error during analysis:\n{str(e)}")
            
    def browse_file(self):
        """Browse untuk file export"""
        formats = {
            "CSV": "CSV Files (*.csv)",
            "Excel": "Excel Files (*.xlsx)",
            "JSON": "JSON Files (*.json)"
        }
        
        selected_format = self.format_combo.currentText()
        file_filter = formats[selected_format]
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Dataset", "", file_filter
        )
        
        if file_path:
            self.file_path_edit.setText(file_path)
            
    def export_dataset(self):
        """Export dataset ke file"""
        if self.current_dataset is None:
            QMessageBox.information(self, "Info", "No dataset available. Please generate dataset first.")
            return
            
        file_path = self.file_path_edit.toPlainText()
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please specify a file path")
            return
            
        try:
            # Convert dataset to DataFrame
            all_data = []
            for cycle in self.current_dataset:
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
            
            # Export based on format
            export_format = self.format_combo.currentText()
            if export_format == "CSV":
                final_df.to_csv(file_path, index=False)
            elif export_format == "Excel":
                final_df.to_excel(file_path, index=False)
            elif export_format == "JSON":
                final_df.to_json(file_path, orient='records', indent=2)
                
            # Update status
            file_size = len(final_df)  # Simplified size calculation
            self.export_status.setText(f"Export successful!\n"
                                     f"File: {file_path}\n"
                                     f"Records: {len(final_df)}\n"
                                     f"Format: {export_format}")
            self.statusBar().showMessage(f"Dataset exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error during export:\n{str(e)}")
            self.export_status.setText(f"Export failed: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = DrivingCycleGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()