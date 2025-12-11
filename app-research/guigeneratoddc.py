import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import seaborn as sns
from scipy import signal
import random
from typing import Dict, List
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QGroupBox, QLabel, QComboBox, QSpinBox, 
                             QDoubleSpinBox, QPushButton, QCheckBox, QTextEdit, 
                             QTabWidget, QProgressBar, QFileDialog, QMessageBox,
                             QSplitter, QScrollArea, QSizePolicy, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

# Import generator class (disederhanakan untuk demo)
class EVDrivingCycleGenerator:
    def __init__(self):
        self.standards = self._define_standard_parameters()
        self.domains = self._define_domain_parameters()
        self.methods = ['markov_chain', 'fourier', 'statistical', 'piecewise_linear', 'hybrid_ensemble']
        
    def _define_standard_parameters(self):
        return {
            'WLTP': {'v_avg': 46.5, 'idle_ratio': 0.13},
            'EPA': {'v_avg': 34.1, 'idle_ratio': 0.18},
            'CLTC': {'v_avg': 28.9, 'idle_ratio': 0.22}
        }
    
    def _define_domain_parameters(self):
        return {
            'urban': {'v_avg_range': (15, 35), 'description': 'Kondisi perkotaan'},
            'suburban': {'v_avg_range': (35, 55), 'description': 'Pinggiran kota'},
            'highway': {'v_avg_range': (70, 90), 'description': 'Jalan tol'},
            'congested': {'v_avg_range': (5, 20), 'description': 'Kemacetan parah'},
            'hilly': {'v_avg_range': (25, 45), 'description': 'Medan berbukit'}
        }
    
    def generate_cycle(self, domain, method, duration=1800):
        # Simple cycle generation for demo
        t = np.linspace(0, duration, duration)
        
        if domain == 'urban':
            base_speed = 25
            variation = 15 * np.sin(0.1*t) + 8 * np.sin(0.5*t)
        elif domain == 'highway':
            base_speed = 80
            variation = 10 * np.sin(0.05*t) + 5 * np.sin(0.2*t)
        elif domain == 'congested':
            base_speed = 15
            variation = 8 * np.sin(0.2*t) + 12 * np.sin(0.8*t)
        else:
            base_speed = 45
            variation = 12 * np.sin(0.08*t) + 6 * np.sin(0.3*t)
        
        speeds = base_speed + variation + np.random.normal(0, 3, duration)
        speeds = np.maximum(0, speeds)
        speeds = np.minimum(130, speeds)
        
        # Add stops for urban/congested
        if domain in ['urban', 'congested']:
            for i in range(len(speeds)):
                if random.random() < 0.02:
                    stop_duration = random.randint(20, 60) if domain == 'congested' else random.randint(10, 30)
                    speeds[i:i+stop_duration] = 0
        
        speeds = signal.savgol_filter(speeds, window_length=51, polyorder=3)
        
        # Calculate acceleration
        accelerations = [0]
        for i in range(1, len(speeds)):
            accel = (speeds[i] - speeds[i-1]) / 3.6
            accelerations.append(accel)
        
        # Add elevation for hilly domain
        if domain == 'hilly':
            elevation = 50 * np.sin(0.03*t) + 30 * np.sin(0.1*t)
            gradient = np.gradient(elevation)
        else:
            elevation = np.zeros(duration)
            gradient = np.zeros(duration)
        
        return {
            'time': t.tolist(),
            'speed_kmh': speeds.tolist(),
            'acceleration_mss': accelerations,
            'elevation_m': elevation.tolist(),
            'gradient_pct': gradient.tolist(),
            'domain': domain,
            'method': method
        }
    
    def calculate_metrics(self, cycle):
        speeds = np.array(cycle['speed_kmh'])
        return {
            'average_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'idle_ratio': np.sum(speeds == 0) / len(speeds),
            'total_distance': np.sum(speeds) / 3600
        }

# Worker thread for generation
class GenerationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, generator, domains, methods, cycles_per_domain, duration):
        super().__init__()
        self.generator = generator
        self.domains = domains
        self.methods = methods
        self.cycles_per_domain = cycles_per_domain
        self.duration = duration
        self.dataset = {'cycles': []}
    
    def run(self):
        try:
            total_cycles = len(self.domains) * len(self.methods) * self.cycles_per_domain
            current_cycle = 0
            
            for domain in self.domains:
                for method in self.methods:
                    for i in range(self.cycles_per_domain):
                        if self.isInterruptionRequested():
                            return
                        
                        cycle = self.generator.generate_cycle(domain, method, self.duration)
                        metrics = self.generator.calculate_metrics(cycle)
                        
                        cycle_data = {
                            'cycle_id': f"{domain}_{method}_{i}",
                            'domain': domain,
                            'method': method,
                            'data': cycle,
                            'metrics': metrics
                        }
                        
                        self.dataset['cycles'].append(cycle_data)
                        current_cycle += 1
                        progress = int((current_cycle / total_cycles) * 100)
                        self.progress.emit(progress)
            
            self.finished.emit(self.dataset)
            
        except Exception as e:
            self.error.emit(str(e))

# Matplotlib Canvas with improved styling
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.style.use('default')
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

# Main Application Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.generator = EVDrivingCycleGenerator()
        self.current_dataset = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('EV Driving Cycle Generator')
        self.setGeometry(100, 100, 1200, 800)  # Reduced size to fit most screens
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        left_panel.setMaximumWidth(350)  # Reduced width
        
        # Right panel - Visualizations
        right_panel = self.create_visualization_panel()
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 850])
        
        main_layout.addWidget(splitter)
        
        # Apply dark theme
        self.apply_dark_theme()
        
    def apply_dark_theme(self):
        # Set dark palette
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        self.setPalette(dark_palette)
        
        # Set style sheet for better widget styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #353535;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #353535;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #fff;
            }
            QPushButton {
                background-color: #2a82da;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a92ea;
            }
            QPushButton:pressed {
                background-color: #1a72ca;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #353535;
                color: white;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                color: white;
                background-color: #353535;
            }
            QProgressBar::chunk {
                background-color: #2a82da;
                width: 10px;
            }
            QTextEdit {
                background-color: #252525;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #353535;
            }
            QTabBar::tab {
                background-color: #454545;
                color: white;
                padding: 8px 16px;
                border: 1px solid #555;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2a82da;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
            QLabel {
                color: white;
            }
        """)
    
    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("EV Driving Cycle Generator")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Domain selection
        domain_group = QGroupBox("Domain Selection")
        domain_layout = QGridLayout(domain_group)
        
        self.domain_checkboxes = {}
        domains = list(self.generator.domains.keys())
        for i, domain in enumerate(domains):
            cb = QCheckBox(domain.title())
            cb.setChecked(True)
            domain_layout.addWidget(cb, i//2, i%2)
            self.domain_checkboxes[domain] = cb
        
        layout.addWidget(domain_group)
        
        # Method selection
        method_group = QGroupBox("Generation Methods")
        method_layout = QVBoxLayout(method_group)
        
        self.method_checkboxes = {}
        for method in self.generator.methods:
            cb = QCheckBox(method.replace('_', ' ').title())
            cb.setChecked(True)
            method_layout.addWidget(cb)
            self.method_checkboxes[method] = cb
        
        layout.addWidget(method_group)
        
        # Parameters
        param_group = QGroupBox("Generation Parameters")
        param_layout = QGridLayout(param_group)
        
        param_layout.addWidget(QLabel("Cycles per Domain:"), 0, 0)
        self.cycles_spin = QSpinBox()
        self.cycles_spin.setRange(1, 100)
        self.cycles_spin.setValue(5)
        param_layout.addWidget(self.cycles_spin, 0, 1)
        
        param_layout.addWidget(QLabel("Duration (seconds):"), 1, 0)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(60, 3600)
        self.duration_spin.setValue(1800)
        self.duration_spin.setSingleStep(60)
        param_layout.addWidget(self.duration_spin, 1, 1)
        
        layout.addWidget(param_group)
        
        # Action buttons
        button_group = QGroupBox("Actions")
        button_layout = QVBoxLayout(button_group)
        
        self.generate_btn = QPushButton("Generate Dataset")
        self.generate_btn.clicked.connect(self.generate_dataset)
        button_layout.addWidget(self.generate_btn)
        
        self.export_btn = QPushButton("Export Dataset")
        self.export_btn.clicked.connect(self.export_dataset)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        self.single_cycle_btn = QPushButton("Generate Single Cycle")
        self.single_cycle_btn.clicked.connect(self.generate_single_cycle)
        button_layout.addWidget(self.single_cycle_btn)
        
        layout.addWidget(button_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to generate")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Domain info
        info_group = QGroupBox("Domain Information")
        info_layout = QVBoxLayout(info_group)
        
        self.domain_info = QTextEdit()
        self.domain_info.setMaximumHeight(120)
        self.domain_info.setReadOnly(True)
        info_layout.addWidget(self.domain_info)
        
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different visualizations
        self.tabs = QTabWidget()
        
        # Cycle visualization tab
        self.cycle_tab = QWidget()
        cycle_layout = QVBoxLayout(self.cycle_tab)
        
        # Cycle selection
        cycle_selection_layout = QHBoxLayout()
        cycle_selection_layout.addWidget(QLabel("Select Cycle:"))
        self.cycle_selector = QComboBox()
        self.cycle_selector.currentTextChanged.connect(self.update_cycle_display)
        cycle_selection_layout.addWidget(self.cycle_selector)
        cycle_selection_layout.addStretch()
        cycle_layout.addLayout(cycle_selection_layout)
        
        # Matplotlib figure for cycle visualization
        self.cycle_canvas = MplCanvas(self, width=8, height=6)
        self.cycle_toolbar = NavigationToolbar(self.cycle_canvas, self)
        cycle_layout.addWidget(self.cycle_toolbar)
        cycle_layout.addWidget(self.cycle_canvas)
        
        self.tabs.addTab(self.cycle_tab, "Cycle Visualization")
        
        # Metrics comparison tab
        self.metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_tab)
        
        self.metrics_canvas = MplCanvas(self, width=8, height=6)
        self.metrics_toolbar = NavigationToolbar(self.metrics_canvas, self)
        metrics_layout.addWidget(self.metrics_toolbar)
        metrics_layout.addWidget(self.metrics_canvas)
        
        self.tabs.addTab(self.metrics_tab, "Metrics Comparison")
        
        # Dataset overview tab
        self.overview_tab = QWidget()
        overview_layout = QVBoxLayout(self.overview_tab)
        
        self.overview_canvas = MplCanvas(self, width=8, height=6)
        self.overview_toolbar = NavigationToolbar(self.overview_canvas, self)
        overview_layout.addWidget(self.overview_toolbar)
        overview_layout.addWidget(self.overview_canvas)
        
        self.tabs.addTab(self.overview_tab, "Dataset Overview")
        
        layout.addWidget(self.tabs)
        
        return panel
    
    def get_selected_domains(self):
        return [domain for domain, cb in self.domain_checkboxes.items() if cb.isChecked()]
    
    def get_selected_methods(self):
        return [method for method, cb in self.method_checkboxes.items() if cb.isChecked()]
    
    def generate_dataset(self):
        domains = self.get_selected_domains()
        methods = self.get_selected_methods()
        
        if not domains:
            QMessageBox.warning(self, "Warning", "Please select at least one domain")
            return
        
        if not methods:
            QMessageBox.warning(self, "Warning", "Please select at least one method")
            return
        
        self.generate_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Generating dataset...")
        
        self.generation_thread = GenerationThread(
            self.generator,
            domains,
            methods,
            self.cycles_spin.value(),
            self.duration_spin.value()
        )
        
        self.generation_thread.progress.connect(self.progress_bar.setValue)
        self.generation_thread.finished.connect(self.on_generation_finished)
        self.generation_thread.error.connect(self.on_generation_error)
        
        self.generation_thread.start()
    
    def on_generation_finished(self, dataset):
        self.current_dataset = dataset
        self.generate_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Generated {len(dataset['cycles'])} cycles successfully!")
        
        # Update cycle selector
        self.cycle_selector.clear()
        for cycle in dataset['cycles']:
            self.cycle_selector.addItem(cycle['cycle_id'])
        
        # Update visualizations
        self.update_all_visualizations()
        
        QMessageBox.information(self, "Success", "Dataset generation completed!")
    
    def on_generation_error(self, error_msg):
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Generation failed")
        QMessageBox.critical(self, "Error", f"Generation failed: {error_msg}")
    
    def generate_single_cycle(self):
        domains = self.get_selected_domains()
        methods = self.get_selected_methods()
        
        if not domains or not methods:
            QMessageBox.warning(self, "Warning", "Please select at least one domain and method")
            return
        
        domain = domains[0]
        method = methods[0]
        
        cycle = self.generator.generate_cycle(domain, method, self.duration_spin.value())
        metrics = self.generator.calculate_metrics(cycle)
        
        # Create a mini-dataset for visualization
        self.current_dataset = {
            'cycles': [{
                'cycle_id': f"{domain}_{method}_single",
                'domain': domain,
                'method': method,
                'data': cycle,
                'metrics': metrics
            }]
        }
        
        self.export_btn.setEnabled(True)
        self.cycle_selector.clear()
        self.cycle_selector.addItem(f"{domain}_{method}_single")
        self.update_all_visualizations()
        
        self.status_label.setText("Single cycle generated")
    
    def update_cycle_display(self):
        if not self.current_dataset or self.cycle_selector.currentText() == "":
            return
        
        cycle_id = self.cycle_selector.currentText()
        cycle_data = next((c for c in self.current_dataset['cycles'] if c['cycle_id'] == cycle_id), None)
        
        if cycle_data:
            self.plot_cycle(cycle_data)
    
    def plot_cycle(self, cycle_data):
        try:
            cycle = cycle_data['data']
            metrics = cycle_data['metrics']
            
            # Clear the figure
            self.cycle_canvas.fig.clear()
            
            time = np.array(cycle['time']) / 60  # Convert to minutes
            speed = np.array(cycle['speed_kmh'])
            acceleration = np.array(cycle['acceleration_mss'])
            elevation = np.array(cycle['elevation_m'])
            
            # Create subplots based on domain
            if cycle['domain'] == 'hilly':
                # Create 3 subplots for hilly domain
                ax1 = self.cycle_canvas.fig.add_subplot(311)
                ax2 = self.cycle_canvas.fig.add_subplot(312, sharex=ax1)
                ax3 = self.cycle_canvas.fig.add_subplot(313, sharex=ax1)
            else:
                # Create 2 subplots for other domains
                ax1 = self.cycle_canvas.fig.add_subplot(211)
                ax2 = self.cycle_canvas.fig.add_subplot(212, sharex=ax1)
            
            # Plot speed
            ax1.plot(time, speed, 'b-', linewidth=1.5, label='Speed')
            ax1.set_ylabel('Speed (km/h)', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            ax1.set_title(f"Driving Cycle: {cycle_data['cycle_id']}\n"
                         f"Avg Speed: {metrics['average_speed']:.1f} km/h | "
                         f"Max Speed: {metrics['max_speed']:.1f} km/h | "
                         f"Idle: {metrics['idle_ratio']:.2f}", fontsize=11)
            
            # Plot acceleration
            ax2.plot(time, acceleration, 'r-', linewidth=1.5, label='Acceleration')
            ax2.set_ylabel('Acceleration (m/s²)', fontsize=10)
            ax2.set_xlabel('Time (minutes)', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            
            # Plot elevation for hilly domain
            if cycle['domain'] == 'hilly':
                ax3.plot(time, elevation, 'g-', linewidth=1.5, label='Elevation')
                ax3.set_ylabel('Elevation (m)', fontsize=10)
                ax3.set_xlabel('Time (minutes)', fontsize=10)
                ax3.grid(True, alpha=0.3)
                ax3.legend(loc='upper right')
            
            self.cycle_canvas.fig.tight_layout()
            self.cycle_canvas.draw()
            
        except Exception as e:
            print(f"Error in plot_cycle: {e}")
            # Create a simple error plot
            self.cycle_canvas.fig.clear()
            ax = self.cycle_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error plotting cycle:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            self.cycle_canvas.draw()
    
    def update_all_visualizations(self):
        if not self.current_dataset:
            return
        
        self.update_cycle_display()
        self.plot_metrics_comparison()
        self.plot_dataset_overview()
    
    def plot_metrics_comparison(self):
        try:
            if not self.current_dataset:
                return
            
            self.metrics_canvas.fig.clear()
            
            # Prepare data for plotting
            domains = []
            methods = []
            avg_speeds = []
            idle_ratios = []
            
            for cycle in self.current_dataset['cycles']:
                domains.append(cycle['domain'])
                methods.append(cycle['method'])
                avg_speeds.append(cycle['metrics']['average_speed'])
                idle_ratios.append(cycle['metrics']['idle_ratio'])
            
            df = pd.DataFrame({
                'Domain': domains,
                'Method': methods,
                'Avg Speed': avg_speeds,
                'Idle Ratio': idle_ratios
            })
            
            # Create subplots
            ax1 = self.metrics_canvas.fig.add_subplot(121)
            ax2 = self.metrics_canvas.fig.add_subplot(122)
            
            # Plot average speed by domain and method
            sns.boxplot(data=df, x='Domain', y='Avg Speed', hue='Method', ax=ax1)
            ax1.set_title('Average Speed Distribution', fontsize=11)
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend(title='Method')
            
            # Plot idle ratio by domain and method
            sns.boxplot(data=df, x='Domain', y='Idle Ratio', hue='Method', ax=ax2)
            ax2.set_title('Idle Ratio Distribution', fontsize=11)
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(title='Method')
            
            self.metrics_canvas.fig.tight_layout()
            self.metrics_canvas.draw()
            
        except Exception as e:
            print(f"Error in plot_metrics_comparison: {e}")
            self.metrics_canvas.fig.clear()
            ax = self.metrics_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error plotting metrics:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            self.metrics_canvas.draw()
    
    def plot_dataset_overview(self):
        try:
            if not self.current_dataset:
                return
            
            self.overview_canvas.fig.clear()
            
            # Count cycles by domain and method
            domain_method_counts = {}
            for cycle in self.current_dataset['cycles']:
                key = (cycle['domain'], cycle['method'])
                domain_method_counts[key] = domain_method_counts.get(key, 0) + 1
            
            # Prepare data for heatmap
            domains = sorted(set([cycle['domain'] for cycle in self.current_dataset['cycles']]))
            methods = sorted(set([cycle['method'] for cycle in self.current_dataset['cycles']]))
            
            heatmap_data = np.zeros((len(domains), len(methods)))
            for i, domain in enumerate(domains):
                for j, method in enumerate(methods):
                    heatmap_data[i, j] = domain_method_counts.get((domain, method), 0)
            
            # Create heatmap
            ax = self.overview_canvas.fig.add_subplot(111)
            
            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            
            # Set labels
            ax.set_xticks(np.arange(len(methods)))
            ax.set_yticks(np.arange(len(domains)))
            ax.set_xticklabels([m.replace('_', ' ').title() for m in methods])
            ax.set_yticklabels([d.title() for d in domains])
            
            # Rotate x labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            for i in range(len(domains)):
                for j in range(len(methods)):
                    text = ax.text(j, i, int(heatmap_data[i, j]),
                                  ha="center", va="center", 
                                  color="white" if heatmap_data[i, j] > np.max(heatmap_data)/2 else "black",
                                  fontsize=10)
            
            ax.set_title("Dataset Overview: Cycle Count by Domain and Method", fontsize=12, pad=20)
            plt.colorbar(im, ax=ax, label='Number of Cycles')
            
            self.overview_canvas.fig.tight_layout()
            self.overview_canvas.draw()
            
        except Exception as e:
            print(f"Error in plot_dataset_overview: {e}")
            self.overview_canvas.fig.clear()
            ax = self.overview_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error plotting overview:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            self.overview_canvas.draw()
    
    def export_dataset(self):
        if not self.current_dataset:
            QMessageBox.warning(self, "Warning", "No dataset to export")
            return
        
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Dataset", "", 
            "CSV Files (*.csv);;JSON Files (*.json);;All Files (*)", 
            options=options
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    # Export metrics as CSV
                    metrics_data = []
                    for cycle in self.current_dataset['cycles']:
                        row = {
                            'cycle_id': cycle['cycle_id'],
                            'domain': cycle['domain'],
                            'method': cycle['method'],
                            **cycle['metrics']
                        }
                        metrics_data.append(row)
                    
                    df = pd.DataFrame(metrics_data)
                    df.to_csv(filename, index=False)
                
                elif filename.endswith('.json'):
                    # Export full dataset as JSON
                    with open(filename, 'w') as f:
                        json.dump(self.current_dataset, f, indent=2, default=str)
                
                else:
                    # Default to CSV
                    filename += '.csv'
                    metrics_data = []
                    for cycle in self.current_dataset['cycles']:
                        row = {
                            'cycle_id': cycle['cycle_id'],
                            'domain': cycle['domain'],
                            'method': cycle['method'],
                            **cycle['metrics']
                        }
                        metrics_data.append(row)
                    
                    df = pd.DataFrame(metrics_data)
                    df.to_csv(filename, index=False)
                
                QMessageBox.information(self, "Success", f"Dataset exported to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()