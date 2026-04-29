import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ====================== Consciousness Dynamics Simulation ======================
# Aligned with Consciousology v7
# Helper code in Section "Boundary Conditions and System Behavior":
# Habit Dynamics with Langevin Noise & RK4


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
    'rho': 0.58,    # Memory formation rate
    'delta': 0.15,  # Forgetting rate
    
    # Langevin Noise Parameters
    'noise_intensity': 0.12,   # Overall noise strength (σ)
    'noise_C': 0.08,           # Noise on dC/dt
    'noise_R': 0.15,           # Noise on dR/dt
    'noise_P': 0.20,           # Noise on dP/dt
    'noise_M': 0.10,           # Noise on dM/dt
}

# ==================== RK4 Integrator with Langevin Noise ====================
def rk4_langevin_step(y, t, dt, p, rng):
    """
    Custom 4th-order Runge-Kutta with additive Langevin noise
    """
    C, R, P, M, F = y
    p = default_params if p is None else p
    
    # Deterministic part
    def deterministic(y):
        C, R, P, M, F = y
        G0, k, v, eta = p['G0'], p['k'], p['v'], p['eta']
        beta, gamma = p['beta'], p['gamma']
        rho_h, delta_h, R_rep = p['rho_h'], p['delta_h'], p['R_rep']
        lambda_, sigma, rho, delta = p['lambda_'], p['sigma'], p['rho'], p['delta']
        
        C_stat = (G0 + k * P + v * C + M) * R - 0.8 * P + eta
        F_happy = beta * C * (1 - np.exp(-k * max(P, 0)))
        
        dC = M * C_stat * np.exp(-gamma * P) if C_stat > 1e-8 else 0.0
        
        Sat = 1 / (1 + np.exp(-0.5 * (F_happy - 2.0)))
        dR = -F_happy + rho_h * (1 - Sat) * R_rep - delta_h * C
        
        dP = -lambda_ * C + sigma + R
        dM = rho * C - delta * M - R
        dF = F_happy
        
        return np.array([dC, dR, dP, dM, dF])
    
    # RK4 steps
    k1 = deterministic(y)
    k2 = deterministic(y + 0.5 * dt * k1)
    k3 = deterministic(y + 0.5 * dt * k2)
    k4 = deterministic(y + dt * k3)
    
    dy_deterministic = (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    # Additive Langevin noise (Gaussian white noise)
    noise = np.zeros(5)
    noise[0] = p['noise_C'] * rng.normal(0, 1)      # Noise on C
    noise[1] = p['noise_R'] * rng.normal(0, 1)      # Noise on R
    noise[2] = p['noise_P'] * rng.normal(0, 1)      # Noise on P
    noise[3] = p['noise_M'] * rng.normal(0, 1)      # Noise on M
    # F is integrated without extra noise
    
    return y + dt * dy_deterministic + np.sqrt(dt) * noise


# ==================== Simulation Function ====================
def simulate_langevin(p=None, t_span=(0, 80), dt=0.05, y0=None, seed=42):
    if p is None:
        p = default_params.copy()
    if y0 is None:
        y0 = np.array([1.2, 2.0, 1.8, 1.1, 0.0])
    
    rng = np.random.default_rng(seed)
    t = np.arange(t_span[0], t_span[1] + dt/2, dt)
    y = np.zeros((len(t), 5))
    y[0] = y0
    
    for i in range(1, len(t)):
        y[i] = rk4_langevin_step(y[i-1], t[i-1], dt, p, rng)
        
        # Prevent negative values (physical constraint)
        y[i, 0] = max(y[i, 0], 0.01)   # C > 0
        y[i, 2] = max(y[i, 2], 0.0)    # P >= 0
    
    return t, y  # t, [C, R, P, M, F]


# ==================== Parameter Sweep ====================
def parameter_sweep(param_name, values, base_params=None, t_span=(0, 60), 
                   title_prefix="Effect of"):
    if base_params is None:
        base_params = default_params.copy()
    
    plt.figure(figsize=(15, 10))
    
    for val in tqdm(values, desc=f"Sweeping {param_name}"):
        p = base_params.copy()
        p[param_name] = val
        
        t, y = simulate_langevin(p, t_span=t_span, seed=42)
        C = y[:, 0]
        
        plt.plot(t, C, label=f"{param_name} = {val:.2f}", linewidth=1.8, alpha=0.85)
    
    plt.title(f"{title_prefix} {param_name} on Consciousness Intensity C")
    plt.xlabel("Time t")
    plt.ylabel("Consciousness Intensity C")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==================== Plot Single Trajectory ====================
def plot_trajectory(t, y, title="Consciousology v7 - Langevin Simulation"):
    C, R, P, M, F = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4]
    
    fig, axs = plt.subplots(3, 2, figsize=(14, 11))
    
    axs[0,0].plot(t, C, 'b-', lw=2.5)
    axs[0,0].set_title(title)
    axs[0,0].set_ylabel('Consciousness C')
    axs[0,0].grid(True, alpha=0.3)
    
    axs[0,1].plot(t, R, 'r-', lw=2)
    axs[0,1].set_ylabel('Resistance R')
    axs[0,1].grid(True, alpha=0.3)
    
    axs[1,0].plot(t, P, 'orange', lw=2)
    axs[1,0].set_ylabel('Pain P')
    axs[1,0].grid(True, alpha=0.3)
    
    axs[1,1].plot(t, M, 'g-', lw=2)
    axs[1,1].set_ylabel('Memory M')
    axs[1,1].grid(True, alpha=0.3)
    
    axs[2,0].plot(t, F, 'purple', lw=2)
    axs[2,0].set_xlabel('Time t')
    axs[2,0].set_ylabel('Accumulated Happiness F')
    axs[2,0].grid(True, alpha=0.3)
    
    axs[2,1].plot(t, C / (P + 1e-6), 'k--', lw=2)
    axs[2,1].set_xlabel('Time t')
    axs[2,1].set_ylabel('C/P Ratio')
    axs[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==================== Main Execution ====================
if __name__ == "__main__":
    print("=== Consciousology v7 - Habit Dynamics with Langevin Noise & RK4 ===\n")
    
    # 1. Single Trajectory with Noise
    print("Running single simulation with Langevin noise...")
    t, y = simulate_langevin(t_span=(0, 70), dt=0.05, seed=123)
    plot_trajectory(t, y, "Scenario 1: Single Trajectory with Langevin Stochastic Noise")
    
    # 2. Parameter Sweep Examples
    print("\nPerforming parameter sweep on 'k' (Pain Transformation Coefficient)...")
    parameter_sweep('k', [0.4, 0.8, 1.2, 1.6], t_span=(0, 55),
                   title_prefix="Effect of Pain Transformation Coefficient")
    
    print("\nPerforming parameter sweep on 'v' (Asymmetry Maintenance)...")
    parameter_sweep('v', [0.05, 0.3, 0.6, 0.9], t_span=(0, 65),
                   title_prefix="Effect of Asymmetry Maintenance Parameter")
    
    print("\nPerforming parameter sweep on 'noise_intensity'...")
    parameter_sweep('noise_intensity', [0.05, 0.15, 0.25, 0.40], t_span=(0, 50),
                   title_prefix="Effect of Overall Noise Intensity")
    
    print("\nAll simulations completed successfully!")
    print("You can now easily analyze oversaturation, heat death, habit inertia,")
    print("and the impact of stochastic fluctuations on consciousness dynamics.")
