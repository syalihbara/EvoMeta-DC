"""
eco_optimization_v2.py
Adaptive Hybrid NSGA3-HBOA v2 vs NSGA-III and HBOA
Benchmarks: DTLZ1-4 (3 objectives)
Outputs: console table with IGD, HV, Spread, Time(s)
Requires: numpy
"""

import numpy as np
import time
import math
from multiprocessing import Pool, cpu_count

RNG = np.random.RandomState(12345)

# ---------------------------
# DTLZ 1-4 (3-objectives)
# ---------------------------
def dtlz1(x, m=3):
    # x: array length nvar
    n = len(x)
    k = n - m + 1
    g = 100 * (k + np.sum((x[m-1:] - 0.5)**2 - np.cos(20 * np.pi * (x[m-1:] - 0.5))))
    f = []
    for i in range(m):
        prod = 0.5 * (1 + g)
        for j in range(m - i - 1):
            prod *= x[j]
        if i > 0:
            prod *= (1 - x[m - i - 1])
        f.append(prod)
    return np.array(f)

def dtlz2(x, m=3):
    n = len(x)
    g = np.sum((x[m-1:] - 0.5)**2)
    f = []
    for i in range(m):
        prod = 1 + g
        for j in range(m - i - 1):
            prod *= math.cos(0.5 * math.pi * x[j])
        if i > 0:
            prod *= math.sin(0.5 * math.pi * x[m - i - 1])
        f.append(prod)
    return np.array(f)

def dtlz3(x, m=3):
    n = len(x)
    k = n - m + 1
    g = 100 * (k + np.sum((x[m-1:] - 0.5)**2 - np.cos(20 * np.pi * (x[m-1:] - 0.5))))
    f = []
    for i in range(m):
        prod = 1 + g
        for j in range(m - i - 1):
            prod *= math.cos(0.5 * math.pi * x[j])
        if i > 0:
            prod *= math.sin(0.5 * math.pi * x[m - i - 1])
        f.append(prod)
    return np.array(f)

def dtlz4(x, m=3, alpha=100):
    n = len(x)
    g = np.sum((x[m-1:] - 0.5)**2)
    f = []
    for i in range(m):
        prod = 1 + g
        for j in range(m - i - 1):
            prod *= math.cos(0.5 * math.pi * (x[j] ** alpha))
        if i > 0:
            prod *= math.sin(0.5 * math.pi * (x[m - i - 1] ** alpha))
        f.append(prod)
    return np.array(f)

# ---------------------------
# Utilities: nondominated sort, crowding distance
# ---------------------------
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

def fast_non_dominated_sort(F):
    # F: array (N, m) objectives (minimization)
    N = F.shape[0]
    S = [[] for _ in range(N)]
    n = np.zeros(N, dtype=int)
    rank = np.zeros(N, dtype=int)
    fronts = []

    for p in range(N):
        for q in range(N):
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            if len(fronts) == 0:
                fronts.append([])
            fronts[0].append(p)

    i = 0
    while i < len(fronts):
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        if Q:
            fronts.append(Q)
    return fronts

def crowding_distance(F, indices):
    # F: (N,m), indices: list of indices to compute distances for
    if len(indices) == 0:
        return np.array([])
    m = F.shape[1]
    dist = np.zeros(len(indices))
    sub = F[indices]
    for j in range(m):
        sorted_idx = np.argsort(sub[:, j])
        fmin = sub[sorted_idx[0], j]
        fmax = sub[sorted_idx[-1], j]
        dist[sorted_idx[0]] = dist[sorted_idx[-1]] = np.inf
        if fmax - fmin == 0:
            continue
        for k in range(1, len(indices)-1):
            dist[sorted_idx[k]] += (sub[sorted_idx[k+1], j] - sub[sorted_idx[k-1], j]) / (fmax - fmin)
    return dist

def select_by_nd_and_crowding(F, population, pop_size):
    fronts = fast_non_dominated_sort(F)
    new_pop = []
    for front in fronts:
        if len(new_pop) + len(front) <= pop_size:
            new_pop.extend(front)
        else:
            remain = pop_size - len(new_pop)
            distances = crowding_distance(F, front)
            # select top remain by distance
            idx_sorted = np.argsort(-distances)
            selected = [front[i] for i in idx_sorted[:remain]]
            new_pop.extend(selected)
            break
    return new_pop

# ---------------------------
# Simple NSGA-III (using nondominated + crowding)
# ---------------------------
def nsga3_simplified(func, n_var=7, n_obj=3, pop_size=80, gen=100):
    pop = RNG.rand(pop_size, n_var)
    start = time.time()
    for g in range(gen):
        # create offspring by adding gaussian noise
        offspring = pop + RNG.normal(0, 0.05, pop.shape)
        offspring = np.clip(offspring, 0, 1)
        combined = np.vstack((pop, offspring))
        F = np.array([func(ind, n_obj) for ind in combined])
        sel_idx = select_by_nd_and_crowding(F, combined, pop_size)
        pop = combined[sel_idx]
    objs = np.array([func(ind, n_obj) for ind in pop])
    return objs, time.time() - start

# ---------------------------
# Honey Badger Optimization (HBOA) - simplified multiobjective via scalarization
# ---------------------------
def hboa_simplified(func, n_var=7, n_obj=3, pop_size=80, gen=100):
    pop = RNG.rand(pop_size, n_var)
    # use random weights for scalarization, update per generation
    start = time.time()
    for g in range(gen):
        weights = RNG.rand(pop_size, n_obj)
        weights = weights / weights.sum(axis=1, keepdims=True)
        vals = np.array([np.dot(weights[i], func(pop[i], n_obj)) for i in range(pop_size)])
        best_idx = np.argmin(vals)
        best = pop[best_idx].copy()
        # motion inspired by HBA
        for i in range(pop_size):
            r = RNG.rand()
            beta = 0.5 + 0.5 * math.sin(2 * math.pi * r)
            A = 2 * r - 1
            D = np.abs(A * (best - pop[i]))
            pop[i] = pop[i] + beta * (best - pop[i]) * r + RNG.normal(0, 0.02, n_var) * D
        pop = np.clip(pop, 0, 1)
    objs = np.array([func(ind, n_obj) for ind in pop])
    return objs, time.time() - start

# ---------------------------
# Hybrid v2: adaptive weighting + partial HBA updates + elite archive
# ---------------------------
def hybrid_v2(func, n_var=7, n_obj=3, pop_size=80, gen=100, elite_frac=0.05):
    pop = RNG.rand(pop_size, n_var)
    archive = []  # store elite decision vectors
    start = time.time()
    elite_count = max(1, int(pop_size * elite_frac))
    for t in range(gen):
        # adaptive HBOA weight (more exploitation later)
        w_hboa = 0.5 + 0.5 * math.cos(math.pi * t / max(1, gen))
        # NSGA-like offspring for majority (60%)
        n_nsga = int(pop_size * 0.6)
        n_hboa = pop_size - n_nsga
        # NSGA offspring: Gaussian perturbation
        parent_idx = RNG.choice(pop_size, n_nsga, replace=True)
        offspring_nsga = pop[parent_idx] + RNG.normal(0, 0.05, (n_nsga, n_var))
        # HBA-inspired updates for remaining (exploitation)
        # pick best so far
        Fpop = np.array([func(ind, n_obj) for ind in pop])
        sums = np.sum(Fpop, axis=1)
        best_idx = np.argmin(sums)
        best = pop[best_idx].copy()
        offspring_hboa = np.empty((n_hboa, n_var))
        for i in range(n_hboa):
            i_idx = RNG.randint(0, pop_size)
            r = RNG.rand()
            A = 2 * r - 1
            beta = w_hboa
            D = np.abs(A * (best - pop[i_idx]))
            step = beta * (best - pop[i_idx]) * r + RNG.normal(0, 0.02, n_var) * D
            offspring_hboa[i] = np.clip(pop[i_idx] + step, 0, 1)
        # new population candidates
        combined = np.vstack((pop, offspring_nsga, offspring_hboa))
        Fcombined = np.array([func(ind, n_obj) for ind in combined])
        # selection by nondominated + crowding
        sel_idx = select_by_nd_and_crowding(Fcombined, combined, pop_size)
        pop = combined[sel_idx]
        # update elite archive
        Fpop = np.array([func(ind, n_obj) for ind in pop])
        idx_sorted = np.argsort(np.sum(Fpop, axis=1))
        elites = [pop[i].copy() for i in idx_sorted[:elite_count]]
        # merge to archive keeping nondominated
        archive.extend(elites)
        archive_np = np.array(archive)
        if archive_np.size > 0:
            Farr = np.array([func(ind, n_obj) for ind in archive_np])
            # keep nondominated from archive
            nd = []
            for i in range(len(archive_np)):
                dominated_flag = False
                for j in range(len(archive_np)):
                    if i != j and dominates(Farr[j], Farr[i]):
                        dominated_flag = True
                        break
                if not dominated_flag:
                    nd.append(i)
            archive = [archive_np[i].copy() for i in nd]
        # limit archive size
        if len(archive) > pop_size * 2:
            archive = archive[:pop_size*2]
    # final local refinement on archive: small gaussian tweaks and keep nondominated
    if len(archive) > 0:
        arch = np.array(archive)
        for _ in range(20):
            cand = arch + RNG.normal(0, 0.01, arch.shape)
            cand = np.clip(cand, 0, 1)
            merged = np.vstack((arch, cand))
            Fm = np.array([func(ind, n_obj) for ind in merged])
            sel_idx = select_by_nd_and_crowding(Fm, merged, min(len(merged), pop_size))
            arch = merged[sel_idx]
        pop = arch[:pop_size]
    objs = np.array([func(ind, n_obj) for ind in pop])
    return objs, time.time() - start

# ---------------------------
# Reference Pareto front generator (approx) by random sampling + nondominated filtering
# ---------------------------
def approximate_true_pf(func, n_var=7, n_obj=3, samples=2000):
    X = RNG.rand(samples, n_var)
    F = np.array([func(x, n_obj) for x in X])
    # nondominated filter
    inds = []
    for i in range(F.shape[0]):
        dominated_flag = False
        for j in range(F.shape[0]):
            if i != j and dominates(F[j], F[i]):
                dominated_flag = True
                break
        if not dominated_flag:
            inds.append(i)
    PF = F[inds]
    # if too many, sample
    if PF.shape[0] > 1000:
        idx = RNG.choice(PF.shape[0], 1000, replace=False)
        PF = PF[idx]
    return PF

# ---------------------------
# Metrics: IGD, Hypervolume, Spread
# ---------------------------
def compute_igd(obtained, true_pf):
    # IGD = mean over true_pf of min distance to obtained
    if true_pf.shape[0] == 0 or obtained.shape[0] == 0:
        return np.inf
    d = np.linalg.norm(true_pf[:, None, :] - obtained[None, :, :], axis=2)  # (true,obt)
    min_d = np.min(d, axis=1)
    return float(np.mean(min_d))

def compute_hv(obtained, ref_point=None):
    # approximate hypervolume for minimization in m dims:
    if obtained.shape[0] == 0:
        return 0.0
    if ref_point is None:
        ref_point = np.max(obtained, axis=0) + 1.0
    # only count points that dominate nothing outside ref (i.e., all f <= ref)
    dominated_vols = []
    for p in obtained:
        if np.any(p > ref_point):
            continue
        vol = np.prod(ref_point - p)
        dominated_vols.append(vol)
    # This simple summation may overcount overlaps; still useful for comparison
    return float(np.sum(dominated_vols))

def compute_spread(obtained):
    if obtained.shape[0] < 2:
        return 0.0
    # sort by first objective then compute distances between consecutive
    idx = np.argsort(obtained[:, 0])
    sorted_pts = obtained[idx]
    dists = np.linalg.norm(np.diff(sorted_pts, axis=0), axis=1)
    if np.mean(dists) == 0:
        return 0.0
    return float(np.std(dists) / np.mean(dists))

# ---------------------------
# Runner for one problem
# ---------------------------
def run_problem(args):
    problem_name, func = args
    n_var = 12  # use 12 decision variables (m-1 + k), typical DTLZ setting
    n_obj = 3
    pop_size = 80
    gen = 120

    # approximate true pareto front
    true_pf = approximate_true_pf(func, n_var=n_var, n_obj=n_obj, samples=5000)

    algos = [
        ("NSGA-III", nsga3_simplified),
        ("HBOA", hboa_simplified),
        ("Hybrid-v2", hybrid_v2)
    ]
    results = []
    for name, algo in algos:
        # run
        objs, elapsed = algo(func, n_var=n_var, n_obj=n_obj, pop_size=pop_size, gen=gen)
        igd = compute_igd(objs, true_pf)
        hv = compute_hv(objs)
        spr = compute_spread(objs)
        # mean objectives for reporting
        mean_obj = np.mean(objs, axis=0)
        results.append({
            "problem": problem_name,
            "algo": name,
            "IGD": igd,
            "HV": hv,
            "Spread": spr,
            "ObjMean": mean_obj,
            "Time": elapsed,
            "Objs": objs
        })
    return results

# ---------------------------
# Main: parallel over problems
# ---------------------------
def main():
    problems = [("DTLZ1", dtlz1), ("DTLZ2", dtlz2), ("DTLZ3", dtlz3), ("DTLZ4", dtlz4)]
    cpu = max(1, cpu_count() - 1)
    with Pool(processes=min(len(problems), cpu)) as pool:
        all = pool.map(run_problem, problems)

    # flatten
    flat = [item for sub in all for item in sub]

    # print nicely per problem
    problems_order = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"]
    print("\n=== ECO-OPTIMIZATION: HYBRID-v2 EXPERIMENT RESULTS ===\n")
    for prob in problems_order:
        print(f"--- {prob} ---")
        rows = [r for r in flat if r["problem"] == prob]
        # header
        header = "{:<12} {:>10} {:>10} {:>10} {:>16} {:>8}".format("Algorithm", "IGD", "HV", "Spread", "ObjMeans (f1,f2,f3)", "Time(s)")
        print(header)
        print("-" * len(header))
        for r in rows:
            objm = r["ObjMean"]
            print("{:<12} {:10.4e} {:10.4f} {:10.4f}   ({:7.4f},{:7.4f},{:7.4f})   {:8.2f}".format(
                r["algo"], r["IGD"], r["HV"], r["Spread"], objm[0], objm[1], objm[2], r["Time"]))
        # best by IGD
        best = min(rows, key=lambda x: x["IGD"])
        print(f">>> Best (IGD) for {prob}: {best['algo']} (IGD={best['IGD']:.4e}, HV={best['HV']:.4f}, Time={best['Time']:.2f}s)\n")

if __name__ == "__main__":
    main()
