# driving_cycle_gui_fixed.py
# Layout fixes for 1366x768: scrollable left panel, responsive plots/tables, spacing/margins
import sys
import os
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
                             QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox,
                             QProgressBar, QMessageBox, QFileDialog, QSplitter,
                             QHeaderView, QFormLayout, QLineEdit, QSizePolicy, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from hybrid_nsga3_hba_optimizer import add_optimization_tab
from performance_index_evaluator import add_performance_index_tab, add_all_optimization_tabs

# Try to import user's generator; fallback to stub for testing if not available
try:
    from synthetic_driving_cycle_generator import DrivingCycleGenerator
except Exception:
    class DrivingCycleGenerator:
        def __init__(self, duration=600, dt=1.0):
            self.duration = duration
            self.dt = dt
            self.vehicle_params = {}
        def generate_dataset_individual_files(self, samples_per_domain, output_dir):
            os.makedirs(output_dir, exist_ok=True)
            domains = ['hilly','urban','suburban','congested','highway']
            meta = []
            for d in domains:
                ddir = os.path.join(output_dir, d)
                os.makedirs(ddir, exist_ok=True)
                for i in range(samples_per_domain):
                    cid = f'{d}_{i:03d}'
                    times = np.arange(0, self.duration, self.dt)
                    vel = np.clip(15 + 5*np.sin(times/30.0) + np.random.randn(len(times))*1.0, 0, 30)
                    df = pd.DataFrame({'time': times, 'velocity': vel})
                    df.to_csv(os.path.join(ddir, f'{cid}.csv'), index=False)
                    meta.append({'cycle_id': cid, 'domain': d,
                                 'battery_Wh_per_km': float(np.random.uniform(100, 350)),
                                 'battery_energy_Wh': float(np.random.uniform(1000,5000)),
                                 'regen_energy_Wh': float(np.random.uniform(10,300)),
                                 'final_soc_percent': float(np.random.uniform(20,100)),
                                 'c_rate_peak': float(np.random.uniform(0.5,3.0)),
                                 'c_rate_avg': float(np.random.uniform(0.2,1.0)),
                                 'avg_speed': float(np.random.uniform(10,50)),
                                 'max_speed': float(np.random.uniform(40,120)),
                                 'distance_km': float(np.random.uniform(1,50)),
                                 'ev_Wh_per_km': float(np.random.uniform(80,400)),
                                 'motor_efficiency_actual': float(np.random.uniform(0.7,0.95)),
                                 'battery_stress_high_power_events': float(np.random.uniform(0,5)),
                                 'soc_consumed_percent': float(np.random.uniform(1,20))})
            pd.DataFrame(meta).to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)


class GenerationThread(QThread):
    """Thread to run generation without freezing GUI"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, generator, samples_per_domain, output_dir):
        super().__init__()
        self.generator = generator
        self.samples_per_domain = samples_per_domain
        self.output_dir = output_dir

    def run(self):
        try:
            # simulate progress quickly
            for i in range(101):
                self.progress.emit(i)
                self.msleep(5)
            # real generation
            self.generator.generate_dataset_individual_files(self.samples_per_domain, self.output_dir)
            self.finished.emit(self.output_dir)
        except Exception as e:
            self.error.emit(str(e))


class MPLCanvas(FigureCanvas):
    """Responsive matplotlib canvas with tight_layout and expanding policy"""
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def plot_cycle(self, time, velocity, domain, cycle_id):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(time, velocity * 3.6, '-', linewidth=1.2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title(f'Driving Cycle: {domain} - {cycle_id}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_battery_analysis(self, df):
        self.fig.clear()
        if df.empty:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            self.draw()
            return
        gs = self.fig.add_gridspec(2, 2)
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax3 = self.fig.add_subplot(gs[1, 0])
        ax4 = self.fig.add_subplot(gs[1, 1])

        domains = df['domain'].unique()
        battery_data = [df[df['domain'] == d]['battery_Wh_per_km'] for d in domains]
        ax1.boxplot(battery_data, labels=domains)
        ax1.set_ylabel('Battery Consumption (Wh/km)')
        ax1.set_title('Battery Consumption by Domain')
        ax1.tick_params(axis='x', rotation=30)

        soc_data = [df[df['domain'] == d]['soc_consumed_percent'] for d in domains]
        ax2.boxplot(soc_data, labels=domains)
        ax2.set_ylabel('SOC Consumed (%)')
        ax2.set_title('State of Charge Consumption')
        ax2.tick_params(axis='x', rotation=30)

        regen_data = [df[df['domain'] == d]['regen_energy_Wh'] for d in domains]
        ax3.boxplot(regen_data, labels=domains)
        ax3.set_ylabel('Regen Energy (Wh)')
        ax3.set_title('Regenerative Braking Energy')
        ax3.tick_params(axis='x', rotation=30)

        crate_data = [df[df['domain'] == d]['c_rate_peak'] for d in domains]
        ax4.boxplot(crate_data, labels=domains)
        ax4.set_ylabel('Peak C-rate')
        ax4.set_title('Battery Peak C-rate by Domain')
        ax4.tick_params(axis='x', rotation=30)

        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_comparison(self, df):
        self.fig.clear()
        if df.empty:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            self.draw()
            return
        gs = self.fig.add_gridspec(2, 2)
        ax1 = self.fig.add_subplot(gs[0, :])
        ax2 = self.fig.add_subplot(gs[1, 0])
        ax3 = self.fig.add_subplot(gs[1, 1])

        domains = df['domain'].unique()
        battery_energy = [df[df['domain'] == d]['battery_energy_Wh'].mean() for d in domains]
        regen_energy = [df[df['domain'] == d]['regen_energy_Wh'].mean() for d in domains]
        net_energy = [b - r for b, r in zip(battery_energy, regen_energy)]

        x = np.arange(len(domains))
        width = 0.25

        ax1.bar(x - width, battery_energy, width, label='Battery Energy')
        ax1.bar(x, regen_energy, width, label='Regen Energy')
        ax1.bar(x + width, net_energy, width, label='Net Energy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(domains)
        ax1.set_title('Energy Flow Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        efficiency = [df[df['domain'] == d]['motor_efficiency_actual'].mean() for d in domains]
        ax2.bar(domains, efficiency)
        ax2.set_ylabel('Motor Efficiency')
        ax2.set_title('Overall Motor Efficiency')
        ax2.tick_params(axis='x', rotation=30)
        ax2.grid(True, alpha=0.3)

        stress_events = [df[df['domain'] == d]['battery_stress_high_power_events'].mean() for d in domains]
        ax3.bar(domains, stress_events)
        ax3.set_ylabel('High Power Events')
        ax3.set_title('Battery Stress Events')
        ax3.tick_params(axis='x', rotation=30)
        ax3.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=2.0)
        self.draw()


class BatteryAnalysisGUI(QMainWindow):
    def __init__(self, preferred_width=1366, preferred_height=768):
        super().__init__()
        self.df = pd.DataFrame()
        self.generator = DrivingCycleGenerator(duration=600, dt=1.0)
        self.current_dataset_path = ""
        self.preferred_width = preferred_width
        self.preferred_height = preferred_height
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Driving Cycle Generator - Battery Analysis (Fixed Layout)")
        win_w = int(self.preferred_width * 0.95)
        win_h = int(self.preferred_height * 0.95)
        self.setGeometry(50, 50, win_w, win_h)

        self.set_dark_theme()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # Left control panel inside a scroll area
        left_panel_widget = self.create_control_panel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_panel_widget)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(520)

        # Right display panel
        right_panel = self.create_display_panel()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, win_w - 420])
        self.optimizer_gui, self.performance_gui = add_all_optimization_tabs(self)
        main_layout.addWidget(splitter)

    def set_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; }
            QGroupBox { font-weight: bold; border: 1px solid #555555; border-radius: 6px; margin-top: 1ex; padding-top: 8px; }
            QGroupBox::title { color: #88c0d0; left: 10px; padding: 0 5px 0 5px; }
            QPushButton { background-color: #4c566a; color: white; border: none; padding: 8px 12px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #5e81ac; } QPushButton:pressed { background-color: #88c0d0; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background-color: #3b4252; color: white; border: 1px solid #555555; border-radius: 3px; padding: 5px; }
            QLabel { color: #eceff4; }
            QProgressBar { border: 1px solid #555555; border-radius: 3px; text-align: center; color: white; }
            QProgressBar::chunk { background-color: #88c0d0; }
            QTableWidget { background-color: #3b4252; color: white; gridline-color: #555555; border: 1px solid #555555; font-size: 10px; }
            QHeaderView::section { background-color: #4c566a; color: white; padding: 6px; border: 1px solid #555555; font-size: 10px; font-weight: bold; }
            QTabWidget::pane { border: 1px solid #555555; background-color: #3b4252; }
            QTabBar::tab { background-color: #4c566a; color: white; padding: 8px 12px; border: 1px solid #555555; }
            QTabBar::tab:selected { background-color: #88c0d0; color: #2e3440; }
        """)

    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_path_edit.setText(directory)

    def create_control_panel(self):
        control_panel = QWidget()
        layout = QVBoxLayout(control_panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Battery Analysis Dashboard")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #88c0d0;")
        layout.addWidget(title)

        # Output directory
        output_group = QGroupBox("Output Directory")
        output_layout = QVBoxLayout(output_group)
        output_path_layout = QHBoxLayout()
        output_path_layout.addWidget(QLabel("Output Path:"))
        self.output_path_edit = QLineEdit(os.path.join(os.getcwd(), "battery_analysis_dataset"))
        output_path_layout.addWidget(self.output_path_edit)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_output_directory)
        output_path_layout.addWidget(self.browse_btn)
        output_layout.addLayout(output_path_layout)
        layout.addWidget(output_group)

        # Vehicle parameters
        vehicle_group = QGroupBox("Vehicle Parameters")
        vehicle_layout = QFormLayout(vehicle_group)
        self.mass_spin = QDoubleSpinBox(); self.mass_spin.setRange(500, 5000); self.mass_spin.setValue(1500); self.mass_spin.setSuffix(" kg")
        vehicle_layout.addRow("Mass:", self.mass_spin)
        self.cd_spin = QDoubleSpinBox(); self.cd_spin.setRange(0.1, 1.0); self.cd_spin.setValue(0.29); self.cd_spin.setDecimals(3)
        vehicle_layout.addRow("Drag coefficient:", self.cd_spin)
        self.area_spin = QDoubleSpinBox(); self.area_spin.setRange(1.0, 5.0); self.area_spin.setValue(2.2); self.area_spin.setSuffix(" m²")
        vehicle_layout.addRow("Frontal area:", self.area_spin)
        layout.addWidget(vehicle_group)

        # Battery params
        battery_group = QGroupBox("Battery Parameters")
        battery_layout = QFormLayout(battery_group)
        self.battery_capacity_spin = QDoubleSpinBox(); self.battery_capacity_spin.setRange(10,200); self.battery_capacity_spin.setValue(50); self.battery_capacity_spin.setSuffix(" kWh")
        battery_layout.addRow("Battery capacity:", self.battery_capacity_spin)
        self.motor_eff_spin = QDoubleSpinBox(); self.motor_eff_spin.setRange(0.1,1.0); self.motor_eff_spin.setValue(0.85); self.motor_eff_spin.setDecimals(3)
        battery_layout.addRow("Motor efficiency:", self.motor_eff_spin)
        self.regen_eff_spin = QDoubleSpinBox(); self.regen_eff_spin.setRange(0.1,1.0); self.regen_eff_spin.setValue(0.70); self.regen_eff_spin.setDecimals(3)
        battery_layout.addRow("Regen efficiency:", self.regen_eff_spin)
        self.aux_power_spin = QDoubleSpinBox(); self.aux_power_spin.setRange(0.1,5.0); self.aux_power_spin.setValue(0.5); self.aux_power_spin.setSuffix(" kW")
        battery_layout.addRow("Auxiliary power:", self.aux_power_spin)
        layout.addWidget(battery_group)

        # Generation params
        gen_group = QGroupBox("Generation Parameters")
        gen_layout = QFormLayout(gen_group)
        self.samples_spin = QSpinBox(); self.samples_spin.setRange(1,1000); self.samples_spin.setValue(20); self.samples_spin.setSuffix(" samples/domain")
        gen_layout.addRow("Samples per domain:", self.samples_spin)
        self.duration_spin = QSpinBox(); self.duration_spin.setRange(60,3600); self.duration_spin.setValue(600); self.duration_spin.setSuffix(" seconds")
        gen_layout.addRow("Cycle duration:", self.duration_spin)
        self.dt_spin = QDoubleSpinBox(); self.dt_spin.setRange(0.1,5.0); self.dt_spin.setValue(1.0); self.dt_spin.setSuffix(" seconds")
        gen_layout.addRow("Time step:", self.dt_spin)
        layout.addWidget(gen_group)

        # Actions
        self.generate_btn = QPushButton("Generate Dataset with Battery Analysis")
        self.generate_btn.clicked.connect(self.generate_dataset)
        layout.addWidget(self.generate_btn)

        self.load_btn = QPushButton("Load Dataset")
        self.load_btn.clicked.connect(self.load_dataset)
        layout.addWidget(self.load_btn)

        self.save_btn = QPushButton("Save Dataset")
        self.save_btn.clicked.connect(self.save_dataset)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Preview
        preview_group = QGroupBox("Cycle Preview")
        preview_layout = QVBoxLayout(preview_group)
        domain_layout = QHBoxLayout()
        domain_layout.addWidget(QLabel("Domain:"))
        self.domain_combo = QComboBox()
        self.domain_combo.addItems(["hilly", "urban", "suburban", "congested", "highway"])
        domain_layout.addWidget(self.domain_combo)
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(0, 999)
        self.sample_spin.setValue(0)
        self.sample_spin.setPrefix("Sample ")
        domain_layout.addWidget(self.sample_spin)
        preview_layout.addLayout(domain_layout)

        self.preview_btn = QPushButton("Preview Cycle")
        self.preview_btn.clicked.connect(self.preview_cycle)
        self.preview_btn.setEnabled(False)
        preview_layout.addWidget(self.preview_btn)
        layout.addWidget(preview_group)

        # Summary
        summary_group = QGroupBox("Battery Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_label = QLabel("No analysis available")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        layout.addWidget(summary_group)

        layout.addStretch()
        return control_panel

    def create_display_panel(self):
        display_panel = QWidget()
        layout = QVBoxLayout(display_panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.tabs = QTabWidget()

        # Table tab
        self.table_tab = QWidget()
        table_layout = QVBoxLayout(self.table_tab)
        table_info = QLabel("Driving cycle indicators with battery analysis")
        table_info.setStyleSheet("color: #88c0d0; font-style: italic; padding: 5px;")
        table_layout.addWidget(table_info)
        self.data_table = QTableWidget()
        self.data_table.setMinimumHeight(220)
        table_layout.addWidget(self.data_table)

        # Battery analysis
        self.battery_tab = QWidget()
        battery_layout = QVBoxLayout(self.battery_tab)
        battery_info = QLabel("Battery performance analysis across domains")
        battery_info.setStyleSheet("color: #88c0d0; font-style: italic; padding: 5px;")
        battery_layout.addWidget(battery_info)
        self.battery_canvas = MPLCanvas(self, width=8, height=5)
        battery_layout.addWidget(self.battery_canvas)
        battery_layout.addStretch()

        # Comparison
        self.comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(self.comparison_tab)
        comparison_info = QLabel("Domain comparison for battery consumption")
        comparison_info.setStyleSheet("color: #88c0d0; font-style: italic; padding: 5px;")
        comparison_layout.addWidget(comparison_info)
        self.comparison_canvas = MPLCanvas(self, width=8, height=5)
        comparison_layout.addWidget(self.comparison_canvas)
        comparison_layout.addStretch()

        # Files
        self.files_tab = QWidget()
        files_layout = QVBoxLayout(self.files_tab)
        files_info = QLabel("Generated files structure:")
        files_info.setStyleSheet("color: #88c0d0; font-style: italic; padding: 5px;")
        files_layout.addWidget(files_info)
        self.files_text = QLabel("No dataset loaded")
        self.files_text.setWordWrap(True)
        self.files_text.setStyleSheet("font-family: monospace; background-color: #3b4252; padding: 10px; border-radius: 5px;")
        files_layout.addWidget(self.files_text)

        # Add tabs
        self.tabs.addTab(self.table_tab, "Data Table")
        self.tabs.addTab(self.battery_tab, "Battery Analysis")
        self.tabs.addTab(self.comparison_tab, "Domain Comparison")
        self.tabs.addTab(self.files_tab, "File Structure")

        layout.addWidget(self.tabs)
        return display_panel

    def generate_dataset(self):
        # Update generator params
        self.generator.duration = self.duration_spin.value()
        self.generator.dt = self.dt_spin.value()
        self.generator.vehicle_params = {
            "mass": self.mass_spin.value(),
            "Cd": self.cd_spin.value(),
            "A": self.area_spin.value(),
            "Crr": 0.015,
            "rho": 1.225,
            "battery_capacity": self.battery_capacity_spin.value(),
            "motor_efficiency": self.motor_eff_spin.value(),
            "regen_efficiency": self.regen_eff_spin.value(),
            "auxiliary_power": self.aux_power_spin.value(),
            "initial_soc": 100
        }

        output_dir = self.output_path_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Warning", "Please specify output directory!")
            return

        self.progress_bar.setVisible(True)
        self.generate_btn.setEnabled(False)

        self.thread = GenerationThread(self.generator, self.samples_spin.value(), output_dir)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_generation_finished)
        self.thread.error.connect(self.on_generation_error)
        self.thread.start()

    def on_generation_finished(self, output_dir):
        self.current_dataset_path = output_dir
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.load_metadata()
        QMessageBox.information(self, "Success",
                                f"Generated dataset with battery analysis!\nOutput directory: {output_dir}\nTotal cycles: {len(self.df)}")

    def on_generation_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Generation failed:\n{error_msg}")

    def load_dataset(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if directory:
            self.current_dataset_path = directory
            self.load_metadata()

    def load_metadata(self):
        if not self.current_dataset_path:
            return
        metadata_path = os.path.join(self.current_dataset_path, "metadata.csv")
        if os.path.exists(metadata_path):
            try:
                self.df = pd.read_csv(metadata_path)
                self.save_btn.setEnabled(True)
                self.preview_btn.setEnabled(True)
                self.update_display()
                QMessageBox.information(self, "Success", f"Loaded {len(self.df)} cycles!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load metadata:\n{str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Metadata file not found in selected directory!")

    def save_dataset(self):
        if not self.current_dataset_path:
            QMessageBox.warning(self, "Warning", "No dataset to save!")
            return
        target_dir = QFileDialog.getExistingDirectory(self, "Select Target Directory")
        if not target_dir:
            return
        try:
            import shutil, datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_dir_name = f"battery_analysis_{timestamp}"
            target_path = os.path.join(target_dir, new_dir_name)
            shutil.copytree(self.current_dataset_path, target_path)
            QMessageBox.information(self, "Success",
                                    f"Dataset saved to:\n{target_path}\n\nTotal files: {self.count_files(target_path)}")
            self.current_dataset_path = target_path
            self.output_path_edit.setText(target_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save dataset:\n{str(e)}")

    def count_files(self, directory):
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len(files)
        return count

    def preview_cycle(self):
        if self.df.empty or not self.current_dataset_path:
            QMessageBox.warning(self, "Warning", "No dataset available!")
            return
        domain = self.domain_combo.currentText()
        sample_num = self.sample_spin.value()
        cycle_id = f"{domain}_{sample_num:03d}"
        csv_path = os.path.join(self.current_dataset_path, domain, f"{cycle_id}.csv")
        if not os.path.exists(csv_path):
            QMessageBox.warning(self, "Warning", f"Cycle file not found: {csv_path}")
            return
        try:
            cycle_df = pd.read_csv(csv_path)
            time = cycle_df['time'].values
            velocity = cycle_df['velocity'].values
            self.battery_canvas.plot_cycle(time, velocity, domain, cycle_id)
            self.tabs.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load cycle file:\n{str(e)}")

    def update_display(self):
        if self.df.empty:
            return
        self.update_table()
        self.update_battery_analysis()
        self.update_comparison()
        self.update_summary()
        self.update_files_structure()

    def update_table(self):
        if self.df.empty:
            return
        battery_columns = ['cycle_id', 'domain', 'battery_Wh_per_km', 'battery_energy_Wh',
                          'regen_energy_Wh', 'final_soc_percent', 'c_rate_peak', 'c_rate_avg']
        other_columns = ['avg_speed', 'max_speed', 'distance_km', 'ev_Wh_per_km']
        display_columns = battery_columns + [col for col in other_columns if col not in battery_columns]
        available_columns = [col for col in display_columns if col in self.df.columns]

        self.data_table.setRowCount(len(self.df))
        self.data_table.setColumnCount(len(available_columns))
        self.data_table.setHorizontalHeaderLabels(available_columns)

        for i, row in self.df.iterrows():
            for j, col in enumerate(available_columns):
                value = row[col]
                if isinstance(value, (int, np.integer)):
                    display_value = str(value)
                elif isinstance(value, (float, np.floating)):
                    if abs(value) < 0.001 or abs(value) > 10000:
                        display_value = f"{value:.4e}"
                    else:
                        display_value = f"{value:.4f}"
                else:
                    display_value = str(value)

                item = QTableWidgetItem(display_value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                if col == 'battery_Wh_per_km':
                    try:
                        numv = float(value)
                        if numv > 300:
                            item.setBackground(QColor('#bf616a'))
                        elif numv < 150:
                            item.setBackground(QColor('#a3be8c'))
                    except Exception:
                        pass
                elif col == 'domain':
                    color_map = {
                        'hilly': '#8fbcbb',
                        'urban': '#a3be8c',
                        'suburban': '#ebcb8b',
                        'congested': '#bf616a',
                        'highway': '#b48ead'
                    }
                    item.setBackground(QColor(color_map.get(value, '#4c566a')))

                self.data_table.setItem(i, j, item)

        # Stretch columns to fill space nicely and avoid overflow
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setSortingEnabled(True)

    def update_battery_analysis(self):
        self.battery_canvas.plot_battery_analysis(self.df)

    def update_comparison(self):
        self.comparison_canvas.plot_comparison(self.df)

    def update_summary(self):
        if self.df.empty:
            self.summary_label.setText("No analysis available")
            return
        total_cycles = len(self.df)
        avg_consumption = self.df['battery_Wh_per_km'].mean()
        avg_regen = self.df['regen_energy_Wh'].mean()
        avg_soc_consumed = self.df['soc_consumed_percent'].mean()
        max_c_rate = self.df['c_rate_peak'].max()

        domain_eff = self.df.groupby('domain')['battery_Wh_per_km'].mean()
        most_efficient_domain = domain_eff.idxmin()
        least_efficient_domain = domain_eff.idxmax()

        summary_text = f"""
        <b>Battery Analysis Summary:</b><br>
        Total cycles analyzed: {total_cycles}<br>
        Average consumption: {avg_consumption:.1f} Wh/km<br>
        Average regeneration: {avg_regen:.1f} Wh<br>
        Average SOC consumed: {avg_soc_consumed:.2f}%<br>
        Maximum C-rate: {max_c_rate:.2f}C<br><br>

        <b>Efficiency by Domain:</b><br>
        Most efficient: {most_efficient_domain} ({domain_eff[most_efficient_domain]:.1f} Wh/km)<br>
        Least efficient: {least_efficient_domain} ({domain_eff[least_efficient_domain]:.1f} Wh/km)<br><br>

        <i>Green: &lt;150 Wh/km, Red: &gt;300 Wh/km</i>
        """
        self.summary_label.setText(summary_text)

    def update_files_structure(self):
        if not self.current_dataset_path or not os.path.exists(self.current_dataset_path):
            self.files_text.setText("No dataset directory available")
            return
        files_structure = f"Dataset Directory: {self.current_dataset_path}\n\n"
        for root, dirs, files in os.walk(self.current_dataset_path):
            level = root.replace(self.current_dataset_path, '').count(os.sep)
            indent = '  ' * level
            files_structure += f"{indent}{os.path.basename(root)}/\n"
            subindent = '  ' * (level + 1)
            for file in files:
                if file.endswith('.csv') or file.endswith('.json'):
                    files_structure += f"{subindent}{file}\n"
        self.files_text.setText(files_structure)


def main():
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = BatteryAnalysisGUI(preferred_width=1366, preferred_height=768)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
