import tkinter as tk
from tkinter import messagebox, ttk
import threading
import numpy as np
from nsga3_hba_core import run_optimization
from objective_functions import N_INDICATORS, N_OBJECTIVES

class NSGA3_HBA_App:
    def __init__(self, master):
        self.master = master
        master.title("NSGA-III-HBA Optimizer (38 Indikator)")
        
        # Variabel untuk Thread dan Stop Event
        self.optimization_thread = None
        self.stop_event = threading.Event()

        # Konfigurasi UI
        main_frame = ttk.Frame(master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input Parameter
        ttk.Label(main_frame, text="Ukuran Populasi (N):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.pop_size_entry = ttk.Entry(main_frame)
        self.pop_size_entry.insert(0, "100")
        self.pop_size_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(main_frame, text="Iterasi Maksimum (t_max):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.max_iter_entry = ttk.Entry(main_frame)
        self.max_iter_entry.insert(0, "50")
        self.max_iter_entry.grid(row=1, column=1, padx=5, pady=5)

        # Informasi Indikator
        ttk.Label(main_frame, 
                  text=f"Input: {N_INDICATORS} Indikator | Tujuan: {N_OBJECTIVES} Objektif",
                  foreground="blue").grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        # Tombol Kontrol
        self.start_button = ttk.Button(main_frame, text="Mulai Optimasi", command=self.start_optimization_thread)
        self.start_button.grid(row=3, column=0, padx=5, pady=10)
        
        self.stop_button = ttk.Button(main_frame, text="Stop", command=self.stop_optimization, state=tk.DISABLED)
        self.stop_button.grid(row=3, column=1, padx=5, pady=10)

        # Log Output
        ttk.Label(main_frame, text="Log Proses:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        
        self.log_text = tk.Text(main_frame, height=15, width=70)
        self.log_text.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        self.log_text.insert(tk.END, "Siap menjalankan algoritma hibrida NSGA-III-HBA.\n")

    def update_log(self, message):
        """Menambahkan pesan ke log GUI. Dipanggil dari thread lain."""
        # Gunakan master.after untuk menjalankan update di main thread
        self.master.after(0, self._append_to_log, message)

    def _append_to_log(self, message):
        """Fungsi internal untuk menambah log di main thread."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END) # Scroll otomatis

    def start_optimization_thread(self):
        """Memvalidasi input dan memulai optimasi dalam thread terpisah."""
        try:
            pop_size = int(self.pop_size_entry.get())
            t_max = int(self.max_iter_entry.get())
            
            if pop_size <= 0 or t_max <= 0:
                 raise ValueError("N dan t_max harus positif.")

            self.update_log("\n--- Memulai ---")
            self.start_button.config(state=tk.DISABLED, text="Sedang Berjalan...")
            self.stop_button.config(state=tk.NORMAL)
            self.stop_event.clear() # Reset stop event
            
            # Buat dan jalankan thread untuk optimasi
            self.optimization_thread = threading.Thread(
                target=self.run_optimization_worker,
                args=(pop_size, t_max)
            )
            self.optimization_thread.start()
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Input harus berupa angka bulat positif. {e}")
            self.start_button.config(state=tk.NORMAL, text="Mulai Optimasi")

    def run_optimization_worker(self, pop_size, t_max):
        """Fungsi pekerja yang menjalankan algoritma core."""
        try:
            # Panggil fungsi inti optimasi
            run_optimization(pop_size, t_max, self.update_log, self.stop_event)
            
        except Exception as e:
            self.update_log(f"FATAL ERROR: {e}")
            messagebox.showerror("Optimasi Error", "Terjadi kesalahan fatal. Lihat log.")
            
        finally:
            # Setelah selesai, aktifkan kembali tombol (di main thread)
            self.master.after(0, self.start_button.config, 
                              {'state': tk.NORMAL, 'text': "Mulai Optimasi"})
            self.master.after(0, self.stop_button.config, {'state': tk.DISABLED})
            
    def stop_optimization(self):
        """Menghentikan optimasi yang sedang berjalan."""
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.stop_event.set()
            self.stop_button.config(state=tk.DISABLED)
            self.update_log("\nPermintaan stop diterima. Menunggu iterasi saat ini selesai...")


if __name__ == '__main__':
    root = tk.Tk()
    app = NSGA3_HBA_App(root)
    root.mainloop()