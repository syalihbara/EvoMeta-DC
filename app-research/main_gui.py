# main_gui.py
import sys, os, threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QTextEdit,
    QHBoxLayout, QSpinBox, QFileDialog, QMessageBox, QProgressDialog, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
import matplotlib.pyplot as plt
import pandas as pd
from optimizer_core import run_hybrid_nsga3_hba

# Worker QObject for running optimizer in background with signals
class OptimizerWorker(QObject):
    progress = pyqtSignal(int, str)       # percent, message
    finished = pyqtSignal(object, object, object)  # final_X, final_F, meta_df
    error = pyqtSignal(str)

    def __init__(self, scenario, pop, gen, save_folder):
        super().__init__()
        self.scenario = scenario
        self.pop = pop
        self.gen = gen
        self.save_folder = save_folder

    def run(self):
        try:
            # define local callback to forward progress
            def cb(percent, message):
                self.progress.emit(int(percent), str(message))
            final_X, final_F, meta_df = run_hybrid_nsga3_hba(pop_size=self.pop, n_gen=self.gen,
                                                              save_folder=self.save_folder, progress_callback=cb)
            self.finished.emit(final_X, final_F, meta_df)
        except Exception as e:
            self.error.emit(str(e))

class DrivingCycleGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synthetic Driving Cycle Optimization (NSGA3 + HBA) — GUI")
        self.setGeometry(120, 80, 1100, 720)
        self._init_ui()
        self.current_results_folder = None

    def _init_ui(self):
        layout = QVBoxLayout()
        top = QHBoxLayout()

        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(["urban", "suburban", "highway", "hilly", "congested"])
        self.pop_spin = QSpinBox(); self.pop_spin.setRange(20, 400); self.pop_spin.setValue(120)
        self.gen_spin = QSpinBox(); self.gen_spin.setRange(10, 500); self.gen_spin.setValue(80)
        self.btn_batch = QPushButton("Generate Batch (placeholder)")
        self.btn_opt = QPushButton("Run NSGA3 + HBA")
        self.btn_view = QPushButton("View Selected Cycle")
        self.btn_view.setEnabled(False)

        top.addWidget(QLabel("Scenario:")); top.addWidget(self.scenario_combo)
        top.addWidget(QLabel("Population:")); top.addWidget(self.pop_spin)
        top.addWidget(QLabel("Generations:")); top.addWidget(self.gen_spin)
        top.addWidget(self.btn_batch); top.addWidget(self.btn_opt); top.addWidget(self.btn_view)

        layout.addLayout(top)

        self.console = QTextEdit(); self.console.setReadOnly(True)
        layout.addWidget(QLabel("Console Output:")); layout.addWidget(self.console, 1)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["File", "Wh/km", "Pmax (kW)", "Jerk RMS", "Battery Stress"])
        layout.addWidget(QLabel("Pareto Solutions (click a row to enable View):"))
        layout.addWidget(self.table, 1)

        self.setLayout(layout)

        # connections
        self.btn_opt.clicked.connect(self.run_optimization)
        self.btn_batch.clicked.connect(self.generate_batch_placeholder)
        self.btn_view.clicked.connect(self.view_selected_cycle)
        self.table.cellClicked.connect(self.on_table_click)

    def log(self, msg):
        self.console.append(msg)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def generate_batch_placeholder(self):
        # In your prior GUI you had batch generation; here we keep placeholder to focus on optimization
        QMessageBox.information(self, "Batch", "Generate batch step (if needed) should be run before optimization.\n(This placeholder assumes optimization can run without manual batch files.)")

    def run_optimization(self):
        scenario = self.scenario_combo.currentText()
        pop = self.pop_spin.value()
        gen = self.gen_spin.value()

        folder = QFileDialog.getExistingDirectory(self, "Select folder to save results (will be created)")
        if not folder:
            return
        results_folder = os.path.join(folder, scenario)
        os.makedirs(results_folder, exist_ok=True)
        self.current_results_folder = results_folder

        # UI: progress dialog
        self.progress_dialog = QProgressDialog("Running optimization...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Optimization Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()

        # Worker + QThread
        self.thread = QThread()
        self.worker = OptimizerWorker(scenario, pop, gen, results_folder)
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
        self.log(f"Started optimization (scenario={scenario}, pop={pop}, gen={gen}), saving to {results_folder}")

    def on_progress(self, percent, message):
        self.progress_dialog.setValue(percent)
        self.progress_dialog.setLabelText(message)
        self.log(f"[{percent}%] {message}")
        QApplication.processEvents()
        if self.progress_dialog.wasCanceled():
            # user canceled: best-effort (no full cancellation implemented inside optimizer)
            self.log("User requested cancellation — optimizer will finish current step then stop (best-effort).")

    def on_finished(self, final_X, final_F, meta_df):
        self.progress_dialog.setValue(100)
        self.log("Optimization finished. Populating table...")
        # populate table with meta_df
        self.table.setRowCount(0)
        for idx, row in meta_df.iterrows():
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(row["file"]))
            self.table.setItem(r, 1, QTableWidgetItem(f"{row['ev_Wh_per_km']:.1f}"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{row['peak_bat_power_kW']:.2f}"))
            self.table.setItem(r, 3, QTableWidgetItem(f"{row['jerk_rms']:.4f}"))
            self.table.setItem(r, 4, QTableWidgetItem(f"{row['battery_stress_proxy']:.1f}"))
        self.log(f"Saved {len(meta_df)} Pareto cycles to {self.current_results_folder}")
        self.thread.quit()
        self.thread.wait()

    def on_error(self, errmsg):
        QMessageBox.critical(self, "Optimization Error", errmsg)
        self.log("Error during optimization: " + errmsg)
        self.thread.quit()
        self.thread.wait()

    def on_table_click(self, row, col):
        # enable view button when a row is selected
        self.btn_view.setEnabled(True)

    def view_selected_cycle(self):
        sel = self.table.currentRow()
        if sel < 0:
            QMessageBox.warning(self, "Select row", "Please select a row first.")
            return
        fname = self.table.item(sel, 0).text()
        fullpath = os.path.join(self.current_results_folder, fname)
        if not os.path.exists(fullpath):
            QMessageBox.warning(self, "File not found", f"{fullpath} not found.")
            return
        df = pd.read_csv(fullpath)
        plt.figure(figsize=(8,4))
        plt.plot(df["time_s"], df["speed_kph"], linewidth=1.2)
        plt.xlabel("Time (s)"); plt.ylabel("Speed (km/h)")
        plt.title(f"Cycle: {fname}")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = DrivingCycleGUI()
    w.show()
    sys.exit(app.exec_())
