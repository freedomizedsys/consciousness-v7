import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pymc as pm
import arviz as az
import seaborn as sns

# ====================== 核心動態系統 ======================
def consciousness_system(t, y, params):
    C, M, P = y
    k, v, alpha, gamma, lambda_mem, W, K, Gp, R, f, H, beta = params
    
    dC_dt = M * (W * K * Gp / R) * np.exp(-gamma * P) * (1 + alpha * M) * v
    dM_dt = lambda_mem * C
    dP_dt = -k * P                      # 痛苦自然緩解
    
    return [dC_dt, dM_dt, dP_dt]

# ====================== Euler 方法 (基準) ======================
def euler_integrate(params, t_max=100, dt=1.0):
    t = np.arange(0, t_max + dt, dt)
    C, M, P = 0.1, 0.0, params[2]   # 初始條件
    C_list, M_list = [], []
    
    for ti in t:
        C_list.append(C)
        M_list.append(M)
        dC, dM, dP = consciousness_system(ti, [C, M, P], params)
        C += dC * dt
        M += dM * dt
        P += dP * dt
        P = max(P, 0)
    
    return t, np.array(C_list), np.array(M_list)

# ====================== RK4 自適應步長 ======================
def rk4_adaptive(params, t_max=100, atol=1e-6):
    sol = solve_ivp(
        fun=lambda t, y: consciousness_system(t, y, params),
        t_span=[0, t_max],
        y0=[0.1, 0.0, params[2]],
        method='RK45',
        atol=atol,
        rtol=1e-6
    )
    return sol.t, sol.y[0], sol.y[1]   # t, C, M

# ====================== Monte Carlo 參數掃描 ======================
def monte_carlo_scan(n_samples=10000):
    results = []
    for _ in range(n_samples):
        k = np.random.uniform(0.1, 1.0)
        v = np.random.uniform(0.01, 0.1)
        P0 = np.random.uniform(1.0, 8.0)
        params = [k, v, 0.2, 0.05, 0.01, 1.0, 5.0, 0.8, 1.5, 0.5, 1.0, 1.0]
        _, C_final, _ = rk4_adaptive(params, t_max=100)
        results.append({'k': k, 'v': v, 'P0': P0, 'C100': C_final[-1]})
    return results

# ====================== Bayesian 擬合範例 ======================
def bayesian_fitting(observed_data):
    with pm.Model() as model:
        k = pm.Uniform('k', 0.1, 1.0)
        v = pm.Uniform('v', 0.01, 0.1)
        sigma = pm.HalfNormal('sigma', sigma=1.0)
        
        # 簡化似然函數（實際使用時替換為真實數據）
        mu = k * 8.0 * v * 10   # 簡化模型
        C_obs = pm.Normal('C_obs', mu=mu, sigma=sigma, observed=observed_data)
        
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)
    return trace

# ====================== 生成四張圖表 ======================
def generate_figures():
    plt.style.use('seaborn-v0_8')
    fig_params = [
        (0.8, 8.0, 1.0, 'High Pain/High Trans (k=0.8)'),   # Sc1
        (0.2, 1.0, 1.0, 'Low Pain/High Happiness (k=0.2)'), # Sc2
        (0.5, 4.0, 1.5, 'Balanced (k=0.5)'),                # Sc3
        (0.15, 4.0, 3.0, 'High Resistance/Low Trans (k=0.15)') # Sc4
    ]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()
    
    for i, (k, P0, R, title) in enumerate(fig_params):
        params = [k, 0.05, 0.2, 0.05, 0.01, 1.0, 5.0, 0.8, R, 0.5, 1.0, 1.0]
        t, C, M = rk4_adaptive(params)
        
        # 圖 1: C(t)
        axs[0].plot(t, C, label=title)
        # 圖 2: M(t)
        axs[1].plot(t, M, label=title)
        # 圖 3: F(t) 簡化示意
        F = 1.0 * C * (1 - np.exp(-k * (8 - np.minimum(8, P0*np.exp(-0.05*t)))))
        axs[2].plot(t, F, label=title)
    
    axs[0].set_title('Figure 1: Time Evolution of Consciousness Intensity C(t)')
    axs[1].set_title('Figure 2: Memory Accumulation M(t)')
    axs[2].set_title('Figure 3: Natural Emergence of Happiness F(t)')
    
    for ax in axs[:3]:
        ax.legend()
        ax.set_xlabel('Time t')
    
    plt.tight_layout()
    plt.savefig('figures/combined_figures.png', dpi=300, bbox_inches='tight')
    print("四張圖表已生成並儲存至 figures/ 資料夾")

# ====================== 主程式 ======================
if __name__ == "__main__":
    generate_figures()
    print("模擬完成！請檢查 figures/ 資料夾中的圖表。")