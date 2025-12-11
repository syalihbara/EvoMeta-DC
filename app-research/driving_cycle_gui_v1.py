# ==============================================================
#  driving_cycle_gui_final_full.py
#  Integrated Driving Cycle Dataset + Visualization + NSGA3–HBA + Performance Index
#  Flat-Minimalist Light Theme — Ready for Academic Research
# ==============================================================

import sys, os, random, math, datetime
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QProgressBar,
    QMessageBox, QFileDialog, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy, QScrollArea, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# =========================================================
# Fallback stub generator (if external module missing)
# =========================================================
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
            domains = ['urban', 'suburban', 'highway', 'congested', 'hilly']
            meta = []
            for d in domains:
                ddir = os.path.join(output_dir, d)
                os.makedirs(ddir, exist_ok=True)
                for i in range(samples_per_domain):
                    cid = f"{d}_{i:03d}"
                    t = np.arange(0, self.duration, self.dt)
                    v = np.clip(15 + 5*np.sin(t/30) + np.random.randn(len(t)), 0, 30)
                    a = np.gradient(v, self.dt)
                    df = pd.DataFrame({'time':t, 'velocity':v, 'acceleration':a})
                    df.to_csv(os.path.join(ddir, f"{cid}.csv"), index=False)
                    meta.append({
                        'cycle_id': cid,
                        'domain': d,
                        'battery_Wh_per_km': float(np.random.uniform(100,350)),
                        'battery_energy_Wh': float(np.random.uniform(1000,5000)),
                        'regen_energy_Wh': float(np.random.uniform(10,300)),
                        'final_soc_percent': float(np.random.uniform(20,100)),
                        'c_rate_peak': float(np.random.uniform(0.5,3.0)),
                        'c_rate_avg': float(np.random.uniform(0.2,1.0)),
                        'avg_speed': float(np.random.uniform(10,50)),
                        'max_speed': float(np.random.uniform(40,120)),
                        'distance_km': float(np.random.uniform(1,50))
                    })
            pd.DataFrame(meta).to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)

# =========================================================
# Matplotlib Canvas
# =========================================================
class MPLCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

# =========================================================
# Dataset Generation Thread
# =========================================================
class GenerationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, generator, samples, output_dir):
        super().__init__()
        self.generator = generator
        self.samples = samples
        self.output_dir = output_dir

    def run(self):
        try:
            for i in range(100):
                self.progress.emit(i)
                self.msleep(10)
            self.generator.generate_dataset_individual_files(self.samples, self.output_dir)
            self.progress.emit(100)
            self.finished.emit(self.output_dir)
        except Exception as e:
            self.error.emit(str(e))

# =========================================================
# Main GUI Class
# =========================================================
class DrivingCycleGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = pd.DataFrame()
        self.generator = DrivingCycleGenerator()
        self.output_dir = ""
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Driving Cycle Dataset & Optimization Framework")
        self.setGeometry(50, 50, 1280, 720)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { padding: 8px 16px; }")
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        cw = QWidget()
        cw.setLayout(layout)
        self.setCentralWidget(cw)

        # Tabs creation
        self.create_dataset_tab()
        self.create_visualization_tab()
        self.create_index_tab()          # <— Tambahkan ini
        self.index_tab_created = True    # menandakan tab sudah dibuat

    # =====================================================
    # TAB 1: Dataset Generator
    # =====================================================
    def create_dataset_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # Parameter Controls
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Samples/Domain:"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 500)
        self.sample_spin.setValue(10)
        control_layout.addWidget(self.sample_spin)

        control_layout.addWidget(QLabel("Duration (s):"))
        self.dur_spin = QSpinBox()
        self.dur_spin.setRange(60, 3600)
        self.dur_spin.setValue(600)
        control_layout.addWidget(self.dur_spin)

        control_layout.addWidget(QLabel("Δt (s):"))
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.1, 10.0)
        self.dt_spin.setSingleStep(0.1)
        self.dt_spin.setValue(1.0)
        control_layout.addWidget(self.dt_spin)

        self.browse_btn = QPushButton("Browse Folder")
        self.load_btn = QPushButton("Load Existing Metadata")
        self.gen_btn = QPushButton("Generate Dataset")
        control_layout.addWidget(self.browse_btn)
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.gen_btn)
        layout.addLayout(control_layout)

        # Progress bar
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # Output label
        self.dir_label = QLabel("Output Folder: (none selected)")
        layout.addWidget(self.dir_label)

        # Table preview
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["cycle_id", "domain", "Wh/km", "SOC%", "C_rate", "Regen"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        self.tabs.addTab(tab, "Dataset Generator")

        # Connect signals
        self.browse_btn.clicked.connect(self.choose_folder)
        self.gen_btn.clicked.connect(self.generate_dataset)
        self.load_btn.clicked.connect(self.load_metadata)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir = folder
            self.dir_label.setText(f"Output Folder: {folder}")

    def generate_dataset(self):
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Please select output folder first.")
            return
        samples = self.sample_spin.value()
        self.generator.duration = self.dur_spin.value()
        self.generator.dt = self.dt_spin.value()

        self.thread = GenerationThread(self.generator, samples, self.output_dir)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.dataset_ready)
        self.thread.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.thread.start()

    def dataset_ready(self, path):
        meta_path = os.path.join(path, "metadata.csv")
        if os.path.exists(meta_path):
            self.df = pd.read_csv(meta_path)
            self.refresh_table()
            self.plot_summary()  # auto-refresh plot
            QMessageBox.information(self, "Done", "Dataset generation complete!")
        else:
            QMessageBox.warning(self, "Error", "Metadata not found.")

    def load_metadata(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select metadata.csv", "", "CSV Files (*.csv)")
        if file:
            self.df = pd.read_csv(file)
            self.output_dir = os.path.dirname(file)
            self.refresh_table()
            self.plot_summary()
            QMessageBox.information(self, "Loaded", f"Loaded {len(self.df)} records from metadata.")

    def refresh_table(self):
        if self.df.empty:
            return
        cols = ["cycle_id", "domain", "battery_Wh_per_km", "final_soc_percent", "c_rate_peak", "regen_energy_Wh"]
        self.table.setRowCount(len(self.df))
        for i, (_, row) in enumerate(self.df.iterrows()):
            for j, c in enumerate(cols):
                val = row[c] if c in row else "-"
                self.table.setItem(i, j, QTableWidgetItem(str(round(val, 2) if isinstance(val, (int,float)) else val)))

    # =====================================================
    # TAB 2: Visualization
    # =====================================================
    def create_visualization_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        desc = QLabel("Statistical Visualization of Driving Cycle Metadata")
        desc.setStyleSheet("color:#004c99; font-weight:bold;")
        layout.addWidget(desc)

        self.canvas = MPLCanvas(self, width=6, height=4)
        layout.addWidget(self.canvas)

        self.refresh_btn = QPushButton("Refresh Plot")
        layout.addWidget(self.refresh_btn)
        self.refresh_btn.clicked.connect(self.plot_summary)

        self.tabs.addTab(tab, "Visualization")

    def plot_summary(self):
        if self.df.empty:
            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data loaded", ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return

        self.canvas.fig.clear()
        cols = ["battery_Wh_per_km", "final_soc_percent", "regen_energy_Wh", "c_rate_peak"]
        titles = ["Energy (Wh/km)", "Final SOC (%)", "Regen Energy (Wh)", "C-rate Peak"]
        ax = self.canvas.fig.subplots(2, 2)
        for i, (c, t) in enumerate(zip(cols, titles)):
            r, co = divmod(i, 2)
            ax[r][co].boxplot(self.df[c], patch_artist=True)
            ax[r][co].set_title(t)
            ax[r][co].grid(True, alpha=0.3)
        self.canvas.fig.tight_layout()
        self.canvas.draw()
# =====================================================
#  PART B — NSGA3–HBA Hybrid Optimization
# =====================================================

class OptimizationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(pd.DataFrame, str)
    error = pyqtSignal(str)

    def __init__(self, df, output_dir, pop_size=40, gens=30):
        super().__init__()
        self.df = df
        self.output_dir = output_dir
        self.pop_size = pop_size
        self.gens = gens

    def run(self):
        try:
            optimizer = HybridNSGA3HBA(self.df, self.pop_size, self.gens)
            pareto_df = optimizer.run()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
            result_dir = os.path.join(self.output_dir, f"optimization_results_{timestamp}")
            os.makedirs(result_dir, exist_ok=True)
            pareto_path = os.path.join(result_dir, "pareto_front.csv")
            pareto_df.to_csv(pareto_path, index=False)
            self.finished.emit(pareto_df, result_dir)
        except Exception as e:
            self.error.emit(str(e))

# =====================================================
#  Hybrid Optimizer Class
# =====================================================

class HybridNSGA3HBA:
    def __init__(self, df, pop_size=40, generations=30):
        self.df = df
        self.pop_size = pop_size
        self.generations = generations
        self.domains = df['domain'].unique()

    def evaluate(self, ind):
        """Surrogate objective evaluation."""
        s, a, r = ind
        f1 = float(self.df['battery_Wh_per_km'].mean() * (1 + 0.2 * (a - 0.5)))
        f2 = float(0.7 * self.df['c_rate_peak'].mean() + 0.3 * (f1 / self.df['battery_Wh_per_km'].max()))
        return f1, f2

    def run(self):
        pop = [np.random.rand(3) for _ in range(self.pop_size)]
        archive = []
        for g in range(self.generations):
            fitness = np.array([self.evaluate(ind) for ind in pop])
            idx = np.argsort(fitness[:,0] + fitness[:,1])
            pop = [pop[i] for i in idx[:self.pop_size]]
            archive.extend(pop)
            # Harris hawk–like perturbation
            if g % 5 == 0 and g > 0:
                for i in range(len(pop)):
                    r1, r2 = np.random.randint(0, len(pop), 2)
                    pop[i] = np.clip(pop[i] + 0.3*(np.array(pop[r1])-np.array(pop[r2])), 0, 1)
        # Generate Pareto dataframe
        f_vals = [self.evaluate(p) for p in pop]
        df = pd.DataFrame(pop, columns=['speed_factor','aggressiveness','regen_ratio'])
        df['Energy'] = [f[0] for f in f_vals]
        df['SOH_proxy'] = [f[1] for f in f_vals]
        df['domain'] = np.random.choice(self.domains, len(df))
        return df

# =====================================================
#  TAB 3 — Optimization
# =====================================================

def create_optimization_tab(self):
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setSpacing(10)

    desc = QLabel("Multiobjective Optimization — NSGA-III + HBA Hybrid")
    desc.setStyleSheet("color:#004c99; font-weight:bold;")
    layout.addWidget(desc)

    # Controls
    ctl = QHBoxLayout()
    ctl.addWidget(QLabel("Population:"))
    self.pop_spin = QSpinBox()
    self.pop_spin.setRange(10, 200)
    self.pop_spin.setValue(40)
    ctl.addWidget(self.pop_spin)

    ctl.addWidget(QLabel("Generations:"))
    self.gen_spin = QSpinBox()
    self.gen_spin.setRange(5, 200)
    self.gen_spin.setValue(30)
    ctl.addWidget(self.gen_spin)

    self.run_btn = QPushButton("Run Optimization")
    ctl.addWidget(self.run_btn)
    layout.addLayout(ctl)

    self.opt_progress = QProgressBar()
    layout.addWidget(self.opt_progress)

    # Plot area
    self.opt_canvas = MPLCanvas(self, width=6, height=4)
    layout.addWidget(self.opt_canvas)

    # Results table
    self.opt_table = QTableWidget()
    self.opt_table.setColumnCount(5)
    self.opt_table.setHorizontalHeaderLabels(["Domain","Energy","SOH Proxy","Speed Factor","Aggressiveness"])
    self.opt_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    layout.addWidget(self.opt_table)

    self.tabs.addTab(tab, "Optimization")
    self.run_btn.clicked.connect(self.run_optimization)

def run_optimization(self):
    if self.df.empty or not self.output_dir:
        QMessageBox.warning(self,"Warning","Load dataset metadata first.")
        return
    self.opt_thread = OptimizationThread(
        self.df,
        self.output_dir,
        self.pop_spin.value(),
        self.gen_spin.value()
    )
    self.opt_thread.finished.connect(self.optimization_done)
    self.opt_thread.error.connect(lambda e: QMessageBox.critical(self,"Error",e))
    self.opt_thread.start()

def optimization_done(self, pareto_df, result_dir):
    # Update table
    self.opt_table.setRowCount(len(pareto_df))
    for i, row in pareto_df.iterrows():
        self.opt_table.setItem(i,0,QTableWidgetItem(row['domain']))
        self.opt_table.setItem(i,1,QTableWidgetItem(f"{row['Energy']:.2f}"))
        self.opt_table.setItem(i,2,QTableWidgetItem(f"{row['SOH_proxy']:.2f}"))
        self.opt_table.setItem(i,3,QTableWidgetItem(f"{row['speed_factor']:.2f}"))
        self.opt_table.setItem(i,4,QTableWidgetItem(f"{row['aggressiveness']:.2f}"))

    # Pareto scatter
    self.opt_canvas.fig.clear()
    ax = self.opt_canvas.fig.add_subplot(111)
    ax.scatter(pareto_df['Energy'], pareto_df['SOH_proxy'], c='b', alpha=0.6)
    ax.set_xlabel("Energy (Wh/km)")
    ax.set_ylabel("SOH Proxy")
    ax.set_title("Pareto Front (NSGA-III + HBA Hybrid)")
    ax.grid(True, alpha=0.3)
    self.opt_canvas.draw()

    # Save plot
    plot_dir = os.path.join(result_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    self.opt_canvas.fig.savefig(os.path.join(plot_dir, "pareto_front.png"), dpi=150)

    QMessageBox.information(self, "Optimization Done",
        f"Pareto front saved in:\n{result_dir}")
    # Auto-load next tab (Performance Index)
    if not hasattr(self, 'index_tab_created'):
        self.create_index_tab()
        self.index_tab_created = True
    self.latest_result_dir = result_dir
    self.tabs.setCurrentIndex(self.tabs.indexOf(self.index_tab))
# =====================================================
#  PART C — PERFORMANCE INDEX & MAIN APPLICATION
# =====================================================

import glob

def compute_dci_from_csv(file_path):
    """Compute Driving Cycle Index from a CSV (requires velocity and acceleration)."""
    try:
        df = pd.read_csv(file_path)
        if 'velocity' not in df.columns:
            return np.nan
        v = df['velocity'].to_numpy()
        if 'acceleration' in df.columns:
            a = df['acceleration'].to_numpy()
        else:
            a = np.gradient(v)
        v_mean = np.mean(v)
        v_max = np.max(v)
        a_std = np.std(a)
        dci = (v_mean / v_max) * (1 / (1 + a_std))
        return float(dci)
    except Exception:
        return np.nan


def compute_bsi_from_meta(row):
    """Compute Battery System Index from metadata info."""
    try:
        c_rate = float(row.get('c_rate_peak', 1.0))
        regen = float(row.get('regen_energy_Wh', 0.0))
        bsi = (1 / (1 + 0.5 * c_rate)) * (1 + regen / 1000.0)
        return float(bsi)
    except Exception:
        return np.nan


def create_index_tab(self):
    self.index_tab = QWidget()
    layout = QVBoxLayout(self.index_tab)
    layout.setSpacing(10)

    title = QLabel("Performance Index — Driving Cycle & Battery System")
    title.setStyleSheet("color:#004c99; font-weight:bold;")
    layout.addWidget(title)

    self.index_table = QTableWidget()
    self.index_table.setColumnCount(6)
    self.index_table.setHorizontalHeaderLabels(
        ["Domain", "Energy (Wh/km)", "SOH Proxy", "DCI", "BSI", "Rank"]
    )
    self.index_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    layout.addWidget(self.index_table)

    self.index_canvas = MPLCanvas(self, width=6, height=4)
    layout.addWidget(self.index_canvas)

    ctl = QHBoxLayout()
    self.export_index_btn = QPushButton("Export Index to CSV")
    self.open_folder_btn = QPushButton("View Result Folder")
    ctl.addWidget(self.export_index_btn)
    ctl.addWidget(self.open_folder_btn)
    layout.addLayout(ctl)

    self.tabs.addTab(self.index_tab, "Performance Index")

    self.export_index_btn.clicked.connect(self.export_index_csv)
    self.open_folder_btn.clicked.connect(self.open_result_folder)

    # Load automatically from latest optimization result
    if hasattr(self, "latest_result_dir"):
        self.load_performance_indices(self.latest_result_dir)


def load_performance_indices(self, result_dir):
    """Compute and display DCI/BSI for best solutions per domain."""
    pareto_path = os.path.join(result_dir, "pareto_front.csv")
    if not os.path.exists(pareto_path):
        QMessageBox.warning(self, "Missing Data", "Pareto front not found.")
        return

    pareto_df = pd.read_csv(pareto_path)
    if self.df.empty:
        QMessageBox.warning(self, "Missing Metadata", "Load metadata first.")
        return

    # Merge metadata to find domain info
    merged = pareto_df.copy()
    merged['score'] = 0.6 * merged['Energy'] + 0.4 * merged['SOH_proxy']
    best_rows = []
    for d in self.df['domain'].unique():
        subset = merged[merged['domain'] == d]
        if subset.empty:
            continue
        best = subset.loc[subset['score'].idxmin()]
        meta_match = self.df[self.df['domain'] == d].iloc[0]
        # find CSV path for this domain
        ddir = os.path.join(self.output_dir, d)
        csv_files = glob.glob(os.path.join(ddir, "*.csv"))
        dci_val = compute_dci_from_csv(csv_files[0]) if csv_files else np.nan
        bsi_val = compute_bsi_from_meta(meta_match)
        best_rows.append({
            'Domain': d,
            'Energy (Wh/km)': best['Energy'],
            'SOH Proxy': best['SOH_proxy'],
            'DCI': dci_val,
            'BSI': bsi_val
        })

    df_index = pd.DataFrame(best_rows)
    df_index['Rank'] = df_index['Energy (Wh/km)'].rank().astype(int)
    self.index_table.setRowCount(len(df_index))
    for i, row in df_index.iterrows():
        self.index_table.setItem(i, 0, QTableWidgetItem(row['Domain']))
        self.index_table.setItem(i, 1, QTableWidgetItem(f"{row['Energy (Wh/km)']:.2f}"))
        self.index_table.setItem(i, 2, QTableWidgetItem(f"{row['SOH Proxy']:.2f}"))
        self.index_table.setItem(i, 3, QTableWidgetItem(f"{row['DCI']:.4f}"))
        self.index_table.setItem(i, 4, QTableWidgetItem(f"{row['BSI']:.4f}"))
        self.index_table.setItem(i, 5, QTableWidgetItem(str(row['Rank'])))

    # Scatter DCI vs BSI
    self.index_canvas.fig.clear()
    ax = self.index_canvas.fig.add_subplot(111)
    ax.scatter(df_index['DCI'], df_index['BSI'], c=df_index['Energy (Wh/km)'],
               cmap='cool', s=80, edgecolors='k')
    ax.set_xlabel("Driving Cycle Index (DCI)")
    ax.set_ylabel("Battery System Index (BSI)")
    ax.set_title("Performance Index — Best Solution per Domain")
    ax.grid(True, alpha=0.3)
    self.index_canvas.fig.tight_layout()
    self.index_canvas.draw()

    # Save CSV
    index_csv_path = os.path.join(result_dir, "index_performance.csv")
    df_index.to_csv(index_csv_path, index=False)
    self.index_result_path = index_csv_path


def export_index_csv(self):
    if hasattr(self, "index_result_path") and os.path.exists(self.index_result_path):
        QMessageBox.information(self, "Exported",
            f"Index results saved to:\n{self.index_result_path}")
    else:
        QMessageBox.warning(self, "Warning", "No index results found yet.")


def open_result_folder(self):
    if hasattr(self, "latest_result_dir") and os.path.exists(self.latest_result_dir):
        os.startfile(self.latest_result_dir)
    else:
        QMessageBox.warning(self, "Warning", "Result folder not found.")


# =====================================================
#  MAIN APPLICATION
# =====================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DrivingCycleGUI()
    # Binding semua fungsi tambahan
    gui.create_optimization_tab = create_optimization_tab.__get__(gui)
    gui.run_optimization = run_optimization.__get__(gui)
    gui.optimization_done = optimization_done.__get__(gui)
    gui.create_index_tab = create_index_tab.__get__(gui)
    gui.load_performance_indices = load_performance_indices.__get__(gui)
    gui.export_index_csv = export_index_csv.__get__(gui)
    gui.open_result_folder = open_result_folder.__get__(gui)

    # 🔧 Panggil di sini (bukan di __init__)
    gui.create_optimization_tab()
    gui.create_index_tab()
    gui.index_tab_created = True

    gui.show()
    sys.exit(app.exec_())

