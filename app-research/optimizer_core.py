# file: optimizer_core.py
from indicators import compute_indicators
import numpy as np

class Individual:
    def __init__(self, genome, scenario):
        self.genome = genome  # control-points speed
        self.scenario = scenario
        self.indicators = None
        self.objectives = None

    def evaluate(self, params):
        t, v = self.decode_genome(self.genome)
        self.indicators = compute_indicators(v, params["dt"], params, self.scenario)
        # Pilih beberapa untuk objektif
        self.objectives = [
            self.indicators["ev_Wh_per_km"],
            self.indicators["jerk_rms"],
            self.indicators["battery_stress_proxy"]
        ]
        return self.objectives

    def decode_genome(self, genome):
        t = np.linspace(0, len(genome) * 2, len(genome))
        v = np.interp(np.linspace(0, len(genome), 500), np.arange(len(genome)), genome)
        return t, v


def evaluate_population(pop, params):
    for ind in pop:
        ind.evaluate(params)
    return pop
