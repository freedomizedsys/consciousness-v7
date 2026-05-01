import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ====================== Consciousness Dynamics Simulation ======================
# Aligned with Consciousology v7 - Extended Habit Dynamics

# ==================== Parameters ====================
default_params = {
    'G0': 1.0,      # Growth base
    'k': 0.85,      # Pain transformation coefficient
    'v': 0.62,      # Asymmetry maintenance
    'eta': 0.18,    # Negentropy efficiency
    'beta': 2.6,    # Happiness feedback strength
    'gamma': 0.32,  # Pain decay exponent
    
    'rho_h': 0.42,  # Habit reinforcement rate
    'delta_h': 0.28,# Conscious habit-breaking rate
    'R_rep': 1.0,   # Repetition intensity
    
    'lambda_': 0.38,# Conscious pain dissolution rate
    'sigma': 0.35,  # External deterministic disturbance
    
    'delta_P': 0.12,   # Pain natural decay rate δ_P
    'gamma_R': 0.45,   # The coupling strength of drag on P and M γ_R
    
    'rho_M': 0.55,     # Memory formation rate ρ_M
    'delta_M': 0.18,   # Forgetting rate δ_M
    
    'phi0': 0.65,           # Basic correction gain
    'gamma_Sat': 1.2,       # The intensity of global error on gain adjustment
    'Lambda': 1.05,         # Projection constant Λ
    'gamma_h': 0.22,        # Entropy correction coefficient γ_h
    
    # Langevin Noise Parameters
    'noise_intensity': 0.12,
    'noise_C': 0.08,
    'noise_R': 0.15,
    'noise_P': 0.20,
    'noise_M': 0.10,
}

# ==================== RK4 Integrator with Langevin Noise ====================
def rk4_langevin_step(y, t, dt, p, rng, C_history=None):
    """
    RK4 with additive Langevin noise
    """
    C, R, P, M, F = y
    p = default_params if p is None else p
    
    C_obs = C
    r = R / (C + 1e-6)      # Internal order parameter
    C_pred = p['Lambda'] * C * (1 + p['gamma_h'] * np.log(1 + max(r, 0)))
    
    # Global error
    delta_e_global = 0.0
    
    if C_history is not None and len(C_history) > 5:
        delta_e_global = np.abs(np.mean(np.diff(C_history[-10:])))
    
    phi = p['phi0'] * (1 + p['gamma_Sat'] * delta_e_global)
    
    error = np.abs(C_pred - C_obs)
    Sat = 1 - phi * error
    Sat = np.clip(Sat, 0.0, 1.0)
    
    # === Deterministic part ===
    def deterministic(y):
        C, R, P, M, F = y
        G0, k, v, eta = p['G0'], p['k'], p['v'], p['eta']
        beta, gamma = p['beta'], p['gamma']
        rho_h, delta_h, R_rep = p['rho_h'], p['delta_h'], p['R_rep']
        lambda_, sigma = p['lambda_'], p['sigma']
        delta_P, gamma_R = p['delta_P'], p['gamma_R']
        rho_M, delta_M = p['rho_M'], p['delta_M']
        
        C_stat = (G0 + k * P + v * C + M) * R - 0.8 * P + eta
        F_happy = beta * C * (1 - np.exp(-k * max(P, 0)))
        
        dC = M * C_stat * np.exp(-gamma * P) if C_stat > 1e-8 else 0.0
        
        dR = -F_happy + rho_h * (1 - Sat) * R_rep - delta_h * C
        
        dP = -lambda_ * C - delta_P * P + gamma_R * R
        dM = rho_M * C - delta_M * M - gamma_R * R
        
        dF = F_happy
        
        return np.array([dC, dR, dP, dM, dF])
    
    # RK4 steps
    k1 = deterministic(y)
    k2 = deterministic(y + 0.5 * dt * k1)
    k3 = deterministic(y + 0.5 * dt * k2)
    k4 = deterministic(y + dt * k3)
    
    dy_deterministic = (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    # Additive Langevin noise
    noise = np.zeros(5)
    noise[0] = p['noise_C'] * rng.normal(0, 1)
    noise[1] = p['noise_R'] * rng.normal(0, 1)
    noise[2] = p['noise_P'] * rng.normal(0, 1)
    noise[3] = p['noise_M'] * rng.normal(0, 1)
    
    return y + dt * dy_deterministic + np.sqrt(dt) * noise, Sat


# ==================== Simulation Function ====================
def simulate_langevin(p=None, t_span=(0, 80), dt=0.05, y0=None, seed=42):

    if p is None:
        p = default_params.copy()
        
    if y0 is None:
        y0 = np.array([1.2, 2.0, 1.8, 1.1, 0.0])   # C, R, P, M, F
    
    rng = np.random.default_rng(seed)
    t = np.arange(t_span[0], t_span[1] + dt/2, dt)
    y = np.zeros((len(t), 5))
    y[0] = y0
    C_history = [y0[0]]
    
    for i in range(1, len(t)):
        y_new, Sat = rk4_langevin_step(y[i-1], t[i-1], dt, p, rng, C_history)
        y[i] = y_new
        
        # physical constraints
        y[i, 0] = max(y[i, 0], 0.01)   # C > 0
        y[i, 2] = max(y[i, 2], 0.0)    # P >= 0
        
        C_history.append(y[i, 0])
    
    return t, y


# ==================== Parameter Sweep ====================
def parameter_sweep(param_name, values, base_params=None, t_span=(0, 60), 
                   title_prefix="Effect of", seed=42):
    """
    Simultaneous drawing Consciousness C 和 Satisfaction Sat
    """
    if base_params is None:
        base_params = default_params.copy()
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    
    for val in tqdm(values, desc=f"Sweeping {param_name}"):
        p = base_params.copy()
        p[param_name] = val
        
        t, y = simulate_langevin(p=p, t_span=t_span, seed=seed)
        C = y[:, 0]
        
        # calculate Sat and C_pred
        Sat, _ = compute_sat_and_cpred(y, p)
        
        axs[0].plot(t, C, label=f"{param_name} = {val:.3f}", linewidth=1.8, alpha=0.85)
        
        axs[1].plot(t, Sat, label=f"{param_name} = {val:.3f}", linewidth=1.8, alpha=0.85)
    
    axs[0].set_title(f"{title_prefix} {param_name} — Consciousness Intensity C")
    axs[0].set_ylabel("Consciousness Intensity C")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=10)
    
    axs[1].set_title(f"{title_prefix} {param_name} — Satisfaction Sat (Prediction-Correction)")
    axs[1].set_xlabel("Time t")
    axs[1].set_ylabel("Satisfaction Sat")
    axs[1].set_ylim(0, 1.05)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    

def compute_sat_and_cpred(y, p, window=10):
    """
    Calculate Sat and C_pred for the entire trajectory.
    """
    C = y[:, 0]
    R = y[:, 1]
    n = len(C)
    
    Sat = np.zeros(n)
    C_pred = np.zeros(n)
    
    for i in range(n):
        C_obs = C[i]
        r = R[i] / (C_obs + 1e-6)                    # Internal order parameter approximation
        
        C_pred[i] = p['Lambda'] * C_obs * (1 + p['gamma_h'] * np.log(1 + max(r, 0)))
        
        if i >= window:
            delta_e_global = np.abs(np.mean(np.diff(C[i-window:i+1])))
        else:
            delta_e_global = 0.0
        
        phi = p['phi0'] * (1 + p['gamma_Sat'] * delta_e_global)
        error = np.abs(C_pred[i] - C_obs)
        Sat[i] = 1 - phi * error
        Sat[i] = np.clip(Sat[i], 0.0, 1.0)
    
    return Sat, C_pred


# ==================== Plot Single Trajectory ====================
def plot_trajectory(t, y, p=None, title="Consciousology v7 - Extended Habit Dynamics"):
    """
    Simultaneously display C, R, P, M, F, C/P, Sat, C vs C_pred
    """
    if p is None:
        p = default_params.copy()
    
    C, R, P, M, F = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4]
    Sat, C_pred = compute_sat_and_cpred(y, p)
    
    fig, axs = plt.subplots(4, 2, figsize=(15, 14))
    
    # Row 0
    axs[0,0].plot(t, C, 'b-', lw=2.5, label='C (Consciousness)')
    axs[0,0].plot(t, C_pred, 'b--', lw=1.8, alpha=0.8, label='C_pred')
    axs[0,0].set_title(title)
    axs[0,0].set_ylabel('Consciousness Intensity')
    axs[0,0].grid(True, alpha=0.3)
    axs[0,0].legend()
    
    axs[0,1].plot(t, R, 'r-', lw=2)
    axs[0,1].set_ylabel('Resistance R')
    axs[0,1].grid(True, alpha=0.3)
    
    # Row 1
    axs[1,0].plot(t, P, 'orange', lw=2)
    axs[1,0].set_ylabel('Pain P')
    axs[1,0].grid(True, alpha=0.3)
    
    axs[1,1].plot(t, M, 'g-', lw=2)
    axs[1,1].set_ylabel('Memory M')
    axs[1,1].grid(True, alpha=0.3)
    
    # Row 2
    axs[2,0].plot(t, F, 'purple', lw=2)
    axs[2,0].set_ylabel('Accumulated Happiness F')
    axs[2,0].grid(True, alpha=0.3)
    
    axs[2,1].plot(t, C / (P + 1e-6), 'k--', lw=2)
    axs[2,1].set_ylabel('C/P Ratio')
    axs[2,1].grid(True, alpha=0.3)
    
    # Row 3
    axs[3,0].plot(t, Sat, 'teal', lw=2.2)
    axs[3,0].set_xlabel('Time t')
    axs[3,0].set_ylabel('Satisfaction Sat')
    axs[3,0].set_ylim(0, 1.05)
    axs[3,0].grid(True, alpha=0.3)
    
    axs[3,1].plot(t, C, 'b-', lw=2, label='C observed')
    axs[3,1].plot(t, C_pred, 'c--', lw=2, label='C predicted')
    axs[3,1].set_xlabel('Time t')
    axs[3,1].set_ylabel('C vs C_pred')
    axs[3,1].grid(True, alpha=0.3)
    axs[3,1].legend()
    
    plt.tight_layout()
    plt.show()


# ==================== Main Execution ====================
if __name__ == "__main__":
    print("=== Consciousology v7 - Extended Habit Dynamics ===\n")
    
    p = default_params.copy()          # Parameters can be adjusted
    t, y = simulate_langevin(p=p, t_span=(0, 80), dt=0.05, seed=123)
    
    plot_trajectory(t, y, p=p, 
                    title="Extended Habit Dynamics with Prediction-Correction Mechanism")
    
    print("\nPerforming parameter sweep on 'k' (Pain Transformation Coefficient)...")
    parameter_sweep('k', [0.4, 0.8, 1.2, 1.6],
                   base_params=p, t_span=(0, 55),
                   title_prefix="Effect of Pain Transformation Coefficient")
    
    print("\nPerforming parameter sweep on 'v' (Asymmetry Maintenance)...")
    parameter_sweep('v', [0.05, 0.3, 0.6, 0.9],
                   base_params=p, t_span=(0, 65),
                   title_prefix="Effect of Asymmetry Maintenance Parameter")

    print("\nPerforming parameter sweep on 'noise_intensity'...")
    parameter_sweep('noise_intensity', [0.05, 0.15, 0.25, 0.40],
                   base_params=p, t_span=(0, 50),
                   title_prefix="Effect of Overall Noise Intensity")
    
    print("\nSweeping gamma_R (Resistance Coupling Strength)...")
    parameter_sweep('gamma_R', [0.1, 0.3, 0.45, 0.7],
                   base_params=p, t_span=(0, 65),
                   title_prefix="Effect of")
    
    print("\nSweeping phi0 (Satisfaction Correction Gain)...")
    parameter_sweep('phi0', [0.3, 0.65, 1.0, 1.5],
                   base_params=p, t_span=(0, 65),
                   title_prefix="Effect of")

    print("\nSweeping rho_h (Habit Reinforcement Rate)...")
    parameter_sweep('rho_h', [0.2, 0.42, 0.65, 0.9], 
                    base_params=p, t_span=(0, 60),
                    title_prefix="Effect of")
    
    print("\nAll simulations completed!")
