import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from solver import run_simulation, find_reattachment_length, x, y, NX, NY
from visualize import (plot_flow_field, plot_velocity_profiles,
                       plot_reattachment_comparison, plot_wall_shear)


QUICK_MODE = '--quick' in sys.argv

if QUICK_MODE:
    RE_LIST   = [100]
    MAX_ITER  = 10000
    TOL       = 1e-4
    print("\n[QUICK MODE] Running Re=100 only with reduced iterations.\n")
else:
    RE_LIST   = [100, 200, 400, 600]
    MAX_ITER  = 60000
    TOL       = 1e-5
    print(f"\n[FULL STUDY] Running Re = {RE_LIST}\n")

armaly_data = {
    100: 2.8,
    200: 5.0,
    400: 7.5,
    600: 9.0,
}

results = {}
t_total = time.time()

for Re in RE_LIST:
    t0 = time.time()
    u, v, p, fluid = run_simulation(
        Re=Re, max_iter=MAX_ITER, tol=TOL, verbose=True)
    t_sim = time.time() - t0

    xr = find_reattachment_length(u, fluid)
    results[Re] = {'u': u, 'v': v, 'p': p, 'fluid': fluid,
                   'Xr': xr, 'time': t_sim}

    print(f"\n  Generatiang plots for Re = {Re}...")
    plot_flow_field(u, v, p, fluid, Re, x, y)
    plot_velocity_profiles(u, fluid, Re, x, y)
    plot_wall_shear(u, fluid, Re, x, y)

print("\n  Generating validation plot...")
Xr_list = [results[Re]['Xr'] for Re in RE_LIST]
plot_reattachment_comparison(RE_LIST, Xr_list)

print("\n" + "="*62)
print(f"  {'Re':>6}  {'X_r/h (sim)':>12}  {'X_r/h (exp)':>12}  {'Error %':>9}  {'Time':>8}")
print("="*62)
for Re in RE_LIST:
    xr_sim = results[Re]['Xr']
    xr_exp = armaly_data.get(Re, None)
    t_s    = results[Re]['time']
    if xr_sim and xr_exp:
        err = abs(xr_sim - xr_exp) / xr_exp * 100
        print(f"  {Re:>6}  {xr_sim:>12.2f}  {xr_exp:>12.1f}  {err:>8.1f}%  {t_s:>6.1f}s")
    elif xr_sim:
        print(f"  {Re:>6}  {xr_sim:>12.2f}  {'N/A':>12}  {'N/A':>9}  {t_s:>6.1f}s")
    else:
        print(f"  {Re:>6}  {'not found':>12}  {xr_exp or 'N/A':>12}  {'N/A':>9}  {t_s:>6.1f}s")
print("="*62)
print(f"\n  Total runtime: {time.time()-t_total:.1f}s")
print(f"  All plots saved to: results/")
print("\n  ✓ Project complete!\n")
