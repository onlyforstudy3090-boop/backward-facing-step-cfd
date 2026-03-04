import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.framealpha': 0.9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

vel_cmap = LinearSegmentedColormap.from_list(
    'vel', ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'])
div_cmap = 'RdBu_r'

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _draw_step(ax, step_i, step_j, x, y, color='#2d2d2d'):
    """Draw the solid step as a filled rectangle."""
    rect = mpatches.FancyBboxPatch(
        (x[0], y[0]), x[step_i], y[step_j],
        boxstyle="square,pad=0",
        facecolor=color, edgecolor='white', linewidth=1.5, zorder=5)
    ax.add_patch(rect)


def plot_flow_field(u, v, p, fluid, Re, x, y, save=True):
    
    step_j = np.argmin(np.abs(y - 1.0))
    step_i = np.argmin(np.abs(x - 1.0))
    X, Y = np.meshgrid(x, y)

    speed = np.sqrt(u**2 + v**2)
    speed[~fluid] = np.nan
    u_plot = np.where(fluid, u, np.nan)
    v_plot = np.where(fluid, v, np.nan)
    p_plot = np.where(fluid, p, np.nan)

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#1a1a2e')
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    panel_style = dict(facecolor='#16213e', edgecolor='#0f3460', linewidth=1.5)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#16213e')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#0f3460')
        spine.set_linewidth(1.5)

    cf1 = ax1.contourf(X, Y, speed, levels=50, cmap=vel_cmap)
    cbar1 = fig.colorbar(cf1, ax=ax1, fraction=0.015, pad=0.01)
    cbar1.set_label('|U| / U_max', color='white', fontsize=10)
    cbar1.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color='white')

    seed_y = np.linspace(1.05, 1.95, 12)
    seed_x = np.full_like(seed_y, 0.1)
    try:
        ax1.streamplot(x, y, u_plot, v_plot,
                       density=2.5, color='white', linewidth=0.6,
                       arrowsize=0.8, arrowstyle='->', broken_streamlines=True)
    except Exception:
        pass

    _draw_step(ax1, step_i, step_j, x, y, color='#0f3460')
    ax1.set_xlim(0, x[-1])
    ax1.set_ylim(0, y[-1])
    ax1.set_xlabel('x / h', color='white')
    ax1.set_ylabel('y / h', color='white')
    ax1.set_title(f'Velocity Magnitude & Streamlines  |  Re = {Re}',
                  color='white', fontweight='bold', fontsize=13)
    ax1.tick_params(colors='white')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#16213e')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#0f3460'); spine.set_linewidth(1.5)

    lim_u = max(abs(np.nanmin(u_plot)), abs(np.nanmax(u_plot)))
    cf2 = ax2.contourf(X, Y, u_plot, levels=40, cmap=div_cmap,
                       vmin=-lim_u, vmax=lim_u)
    ax2.contour(X, Y, u_plot, levels=[0], colors='yellow', linewidths=1.5,
                linestyles='--')
    cbar2 = fig.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('u / U_max', color='white', fontsize=9)
    cbar2.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')
    _draw_step(ax2, step_i, step_j, x, y, color='#0f3460')
    ax2.set_xlim(0, x[-1]); ax2.set_ylim(0, y[-1])
    ax2.set_xlabel('x / h', color='white'); ax2.set_ylabel('y / h', color='white')
    ax2.set_title('Streamwise Velocity  u', color='white', fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.text(0.02, 0.95, 'Yellow dashed: u=0 (separation/reattachment)',
             transform=ax2.transAxes, color='yellow', fontsize=8, va='top')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#16213e')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#0f3460'); spine.set_linewidth(1.5)

    cf3 = ax3.contourf(X, Y, p_plot, levels=40, cmap='plasma')
    cbar3 = fig.colorbar(cf3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('p / (ρU²)', color='white', fontsize=9)
    cbar3.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar3.ax.yaxis.get_ticklabels(), color='white')
    _draw_step(ax3, step_i, step_j, x, y, color='#1a1a2e')
    ax3.set_xlim(0, x[-1]); ax3.set_ylim(0, y[-1])
    ax3.set_xlabel('x / h', color='white'); ax3.set_ylabel('y / h', color='white')
    ax3.set_title('Pressure Field  p', color='white', fontweight='bold')
    ax3.tick_params(colors='white')

    if save:
        path = os.path.join(OUTPUT_DIR, f'flow_field_Re{Re}.png')
        fig.savefig(path, facecolor=fig.get_facecolor())
        print(f"  Saved: {path}")
    plt.close(fig)
    return fig


def plot_velocity_profiles(u, fluid, Re, x, y, x_locs=None, save=True):
    
    step_i = int(round(1.0 / (x[1]-x[0])))
    step_j = np.argmin(np.abs(y - 1.0))

    if x_locs is None:
        x_locs = [2, 4, 6, 8, 12, 16, 20]

    fig, axes = plt.subplots(1, len(x_locs), figsize=(16, 5), sharey=True)
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(f'Streamwise Velocity Profiles  u(y)  |  Re = {Re}',
                 color='white', fontsize=13, fontweight='bold', y=1.01)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(x_locs)))

    for ax, x_loc, col in zip(axes, x_locs, colors):
        ax.set_facecolor('#16213e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#0f3460'); spine.set_linewidth(1.2)

        i_idx = np.argmin(np.abs(x - x_loc))
        u_prof = u[:, i_idx].copy()
        u_prof[:step_j-1] = np.nan
        y_prof = y.copy()
        fluid_mask = fluid[:, i_idx]
        u_prof[~fluid_mask] = np.nan

        ax.plot(u_prof, y_prof, color=col, linewidth=2)
        ax.axvline(0, color='yellow', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_xlabel('u / U_max', color='white', fontsize=9)
        ax.set_title(f'x/h = {x_loc}', color='white', fontsize=10, fontweight='bold')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_xlim(-0.4, 1.2)
        ax.set_ylim(0, y[-1])
        if x_loc <= 1.0:
            ax.axhspan(0, 1.0, color='#0f3460', alpha=0.8)

    axes[0].set_ylabel('y / h', color='white')

    if save:
        path = os.path.join(OUTPUT_DIR, f'velocity_profiles_Re{Re}.png')
        fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_reattachment_comparison(Re_list, Xr_sim, save=True):
    armaly_Re  = [75,  100, 150, 200, 300, 400, 500, 600, 800]
    armaly_Xr  = [2.0, 2.8, 3.8, 5.0, 6.5, 7.5, 8.2, 9.0, 10.2]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#0f3460'); spine.set_linewidth(1.5)

    ax.plot(armaly_Re, armaly_Xr, 'o--', color='#fca636',
            linewidth=2, markersize=8, label='Armaly et al. (1983) — Experiment',
            zorder=4)

    valid = [(re, xr) for re, xr in zip(Re_list, Xr_sim) if xr is not None]
    if valid:
        re_v, xr_v = zip(*valid)
        ax.plot(re_v, xr_v, 's-', color='#f0f921',
                linewidth=2.5, markersize=10, label='Present Simulation (Python FD)',
                zorder=5)
        for re_i, xr_i in zip(re_v, xr_v):
            ax.annotate(f'  {xr_i:.1f}', (re_i, xr_i), color='#f0f921',
                        fontsize=9, va='center')

    ax.set_xlabel('Reynolds Number  Re', color='white', fontsize=12)
    ax.set_ylabel('Reattachment Length  X_r / h', color='white', fontsize=12)
    ax.set_title('Reattachment Length vs Reynolds Number\nValidation against Armaly et al. (1983)',
                 color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(True, color='#0f3460', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.legend(facecolor='#16213e', edgecolor='#0f3460', labelcolor='white',
              fontsize=10, loc='upper left')

    ax.set_xlim(0, max(max(armaly_Re), max(Re_list) if Re_list else 100) * 1.1)
    valid_xr = [xr for xr in Xr_sim if xr is not None]
    sim_max = max(valid_xr) if valid_xr else 0
    ax.set_ylim(0, max(max(armaly_Xr), sim_max) * 1.3)

    ax.text(0.98, 0.05,
            'Expansion ratio ER = 2:1\nStep height h = 1 (reference length)',
            transform=ax.transAxes, color='#a0a0c0', fontsize=9,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f3460',
                      edgecolor='#6060a0', alpha=0.8))

    if save:
        path = os.path.join(OUTPUT_DIR, 'reattachment_validation.png')
        fig.savefig(path, facecolor=fig.get_facecolor())
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_wall_shear(u, fluid, Re, x, y, save=True):
    step_i = int(round(1.0 / (x[1]-x[0])))
    dy = y[1] - y[0]
    skip = 5
    x_wall = x[step_i + skip:]
    tau_w = u[1, step_i + skip:] / dy 
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#0f3460'); spine.set_linewidth(1.5)

    ax.plot(x_wall, tau_w, color='#b12a90', linewidth=2.5)
    ax.axhline(0, color='yellow', linestyle='--', linewidth=1.2, alpha=0.8,
               label='τ_w = 0 (separation / reattachment)')
    ax.fill_between(x_wall, tau_w, 0,
                    where=(tau_w < 0), alpha=0.3, color='#0d0887',
                    label='Recirculation zone (τ_w < 0)')
    ax.fill_between(x_wall, tau_w, 0,
                    where=(tau_w >= 0), alpha=0.3, color='#fca636',
                    label='Attached flow (τ_w > 0)')

    ax.set_xlabel('x / h', color='white', fontsize=12)
    ax.set_ylabel('τ_w / (μU/h)', color='white', fontsize=12)
    ax.set_title(f'Wall Shear Stress along Bottom Wall  |  Re = {Re}',
                 color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(True, color='#0f3460', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(facecolor='#16213e', edgecolor='#0f3460', labelcolor='white', fontsize=9)
    ax.set_ylim(-1.0, 2.0)

    if save:
        path = os.path.join(OUTPUT_DIR, f'wall_shear_Re{Re}.png')
        fig.savefig(path, facecolor=fig.get_facecolor())
        print(f"  Saved: {path}")
    plt.close(fig)
