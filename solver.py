import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import os
import time

L  = 30.0  
H  = 2.0    
Xs = 1.0    
h  = 1.0    


NX = 300   
NY = 40     

dx = L / (NX - 1)
dy = H / (NY - 1)

x = np.linspace(0, L, NX)
y = np.linspace(0, H, NY)
X, Y = np.meshgrid(x, y)  

def run_simulation(Re, max_iter=50000, dt=None, tol=1e-5, verbose=True):
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"  Solving for Re = {Re}")
        print(f"  Grid: {NX} x {NY},  dx={dx:.4f}, dy={dy:.4f}")
        print(f"{'='*50}")

    if dt is None:
        U_max = 1.0
        dt_cfl  = 0.5 * min(dx, dy) / (U_max + 1e-10)
        dt_diff = 0.25 * Re * min(dx, dy)**2   
        dt = 0.4 * min(dt_cfl, dt_diff)
    if verbose:
        print(f"  dt = {dt:.6f}")

    u = np.zeros((NY, NX))   
    v = np.zeros((NY, NX))  
    p = np.zeros((NY, NX))   

    fluid = np.ones((NY, NX), dtype=bool)
    step_j = int(round(h / dy))
    step_i = int(round(Xs / dx))
    fluid[:step_j, :step_i] = False

    u = np.zeros((NY, NX))
    v = np.zeros((NY, NX))
    p = np.zeros((NY, NX))

    y_lo = y[step_j]
    y_hi = y[NY - 1]
    for j_idx in range(step_j, NY):
        u_val = 4.0 * (y[j_idx] - y_lo) * (y_hi - y[j_idx]) / (y_hi - y_lo) ** 2
        u[j_idx, :] = u_val
    u[~fluid] = 0.0

    def apply_bc(u, v):
        j_in_lo = step_j  
        j_in_hi = NY - 1  
        y_in = y[j_in_lo:j_in_hi+1]
        y_lo = y[j_in_lo]
        y_hi = y[j_in_hi]
        u_profile = 4.0 * (y_in - y_lo) * (y_hi - y_in) / (y_hi - y_lo)**2
        u[j_in_lo:j_in_hi+1, 0] = u_profile
        v[j_in_lo:j_in_hi+1, 0] = 0.0

        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]

        u[-1, :] = 0.0
        v[-1, :] = 0.0

        u[0, step_i:] = 0.0
        v[0, step_i:] = 0.0

        u[:step_j, step_i] = 0.0
        v[:step_j, step_i] = 0.0

        u[step_j, :step_i+1] = 0.0
        v[step_j, :step_i+1] = 0.0

        u[~fluid] = 0.0
        v[~fluid] = 0.0

        return u, v

    u, v = apply_bc(u, v)

    t_start = time.time()
    for it in range(max_iter):
        u_old = u.copy()
        v_old = v.copy()

        u_star = u.copy()
        v_star = v.copy()

       
        i = np.arange(1, NX-1)
        j = np.arange(1, NY-1)
        jj, ii = np.meshgrid(j, i, indexing='ij')

        mask = fluid[jj, ii]


        adv_u = (u[jj, ii] * np.where(u[jj,ii]>0,
                    (u[jj,ii]-u[jj,ii-1])/dx,
                    (u[jj,ii+1]-u[jj,ii])/dx) +
                 v[jj, ii] * np.where(v[jj,ii]>0,
                    (u[jj,ii]-u[jj-1,ii])/dy,
                    (u[jj+1,ii]-u[jj,ii])/dy))
        diff_u = ((u[jj,ii+1] - 2*u[jj,ii] + u[jj,ii-1])/dx**2 +
                  (u[jj+1,ii] - 2*u[jj,ii] + u[jj-1,ii])/dy**2) / Re


        adv_v = (u[jj, ii] * np.where(u[jj,ii]>0,
                    (v[jj,ii]-v[jj,ii-1])/dx,
                    (v[jj,ii+1]-v[jj,ii])/dx) +
                 v[jj, ii] * np.where(v[jj,ii]>0,
                    (v[jj,ii]-v[jj-1,ii])/dy,
                    (v[jj+1,ii]-v[jj,ii])/dy))
        diff_v = ((v[jj,ii+1] - 2*v[jj,ii] + v[jj,ii-1])/dx**2 +
                  (v[jj+1,ii] - 2*v[jj,ii] + v[jj-1,ii])/dy**2) / Re

        u_star[jj, ii] = np.where(mask, u[jj,ii] + dt*(-adv_u + diff_u), 0.0)
        v_star[jj, ii] = np.where(mask, v[jj,ii] + dt*(-adv_v + diff_v), 0.0)

        u_star, v_star = apply_bc(u_star, v_star)

        
        rhs = np.zeros((NY, NX))
        rhs[jj, ii] = np.where(mask,
            (u_star[jj,ii+1]-u_star[jj,ii-1])/(2*dx) +
            (v_star[jj+1,ii]-v_star[jj-1,ii])/(2*dy), 0.0) / dt

       
        p_new = p.copy()
        for _ in range(50):
            p_new[jj, ii] = np.where(mask,
                ((p_new[jj,ii+1]+p_new[jj,ii-1])*dy**2 +
                 (p_new[jj+1,ii]+p_new[jj-1,ii])*dx**2 -
                 rhs[jj,ii]*dx**2*dy**2) / (2*(dx**2+dy**2)),
                p_new[jj, ii])
           
            p_new[:, -1] = 0.0             
            p_new[:, 0]  = p_new[:, 1]
            p_new[-1, :] = p_new[-2, :]
            p_new[0, step_i:] = p_new[1, step_i:]

        p = p_new

       
        u[jj, ii] = np.where(mask,
            u_star[jj,ii] - dt*(p[jj,ii+1]-p[jj,ii-1])/(2*dx), 0.0)
        v[jj, ii] = np.where(mask,
            v_star[jj,ii] - dt*(p[jj+1,ii]-p[jj-1,ii])/(2*dy), 0.0)

        u, v = apply_bc(u, v)

        
        du = np.max(np.abs(u - u_old))
        if it % 500 == 0 and verbose:
            elapsed = time.time() - t_start
            print(f"  iter {it:6d} | max|Δu| = {du:.2e} | t = {elapsed:.1f}s")
        if du < tol and it > 100:
            if verbose:
                print(f"\n  ✓ Converged at iteration {it} (max|Δu| = {du:.2e})")
            break
    else:
        if verbose:
            print(f"\n  ⚠ Reached max iterations ({max_iter})")

    return u, v, p, fluid


def find_reattachment_length(u, fluid):
    
    step_i = int(round(Xs / dx))

    tau = u[1, step_i:] / dy

    skip = 5
    ws = tau[skip:]

    for i_idx in range(1, len(ws)):
        if ws[i_idx-1] < 0 and ws[i_idx] >= 0:
            return x[step_i + skip + i_idx] / h

    if np.any(ws < 0):
        return x[step_i + skip + np.where(ws < 0)[0][-1]] / h

    return None


if __name__ == "__main__":
    u, v, p, fluid = run_simulation(Re=100, max_iter=20000, tol=1e-5)
    xr = find_reattachment_length(u, fluid)
    print(f"\nReattachment length X_r/h = {xr:.2f}")
    print("Solver test complete.")
