import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pymc as pm
import arviz as az
import seaborn as sns

# ================= Euler method (benchmark) =================
# ====================== 歐拉方法（基準）======================
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

# ================== RK4 Adaptive Step Size =================
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

# ================= Monte Carlo parameter scan ================
# ====================== 蒙特卡羅參數掃描 ======================
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

# ================ Bayesian Fitting Paradigm ================
# ====================== 貝葉斯擬合範式 ======================
def bayesian_fitting(observed_data):
    with pm.Model() as model:
        k = pm.Uniform('k', 0.1, 1.0)
        v = pm.Uniform('v', 0.01, 0.1)
        sigma = pm.HalfNormal('sigma', sigma=1.0)

        # Simplified likelihood function (replace with real data in actual use)
        # 簡化似然函數（實際使用時替換為真實數據）
        mu = k * 8.0 * v * 10   # Simplified model 簡化模型
        C_obs = pm.Normal('C_obs', mu=mu, sigma=sigma, observed=observed_data)
        
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)
    return trace

# ================== Core Dynamic System ==================
# ====================== 核心動態系統 ======================
def consciousness_system(t, y, params):
    C, M, P = y
    k, v, alpha, gamma, lambda_mem, W, K, Gp, R, f, H, beta = params
    
    dC_dt = M * (W * K * Gp / R) * np.exp(-gamma * P) * (1 + alpha * M) * v
    dM_dt = lambda_mem * C
    dP_dt = -k * P                      # Natural pain relief 自然緩解痛苦
    
    return [dC_dt, dM_dt, dP_dt]

# ================= RK4 Adaptive Step Size ==================
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

# ================= Generate four charts ==================
# ====================== 生成四張圖表 ======================
def generate_figures():
    plt.style.use('seaborn-v0_8')
    
    # Four context parameters 四種情境參數
    scenarios = [
        (0.8, 8.0, 1.0, 'Sc1: 高痛苦高轉化 (k=0.8)', 'High Pain/High Trans (k=0.8)'),
        (0.2, 1.0, 1.0, 'Sc2: 低痛苦高快樂 (k=0.2)', 'Low Pain/High Happiness (k=0.2)'),
        (0.5, 4.0, 1.5, 'Sc3: 平衡情境 (k=0.5)', 'Balanced (k=0.5)'),
        (0.15, 4.0, 3.0, 'Sc4: 高阻力低轉化 (k=0.15)', 'High Resistance/Low Trans (k=0.15)')
    ]
    
    # Create a 2x2 subgraph 建立 2x2 子圖
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    axs = axs.ravel()
    
    for i, (k, P0, R, label_zh, label_en) in enumerate(scenarios):
        params = [k, 0.05, 0.2, 0.05, 0.01, 1.0, 5.0, 0.8, R, 0.5, 1.0, 1.0]
        t, C, M = rk4_adaptive(params)
        
        # Figure 1 / 圖 1: C(t) 意識強度演化
        axs[0].plot(t, C, label=label_zh)
        
        # Figure 2 / 圖 2: M(t) 記憶累積
        axs[1].plot(t, M, label=label_zh)
        
        # Figure 3 / 圖 3: F(t) 快樂自然湧現（簡化計算）
        F = 1.0 * C * (1 - np.exp(-k * (8.0 - np.minimum(8.0, P0 * np.exp(-0.05 * t)))))
        axs[2].plot(t, F, label=label_zh)
    
    # Set chart title and legend 設定圖表標題與圖例
    axs[0].set_title('圖 1：意識強度 C(t) 的時間演化曲線\nFigure 1: Time Evolution of Consciousness Intensity C(t)')
    axs[1].set_title('圖 2：長期記憶強度 M(t) 的累積曲線\nFigure 2: Memory Accumulation M(t)')
    axs[2].set_title('圖 3：快樂 F(t) 的自然湧現曲線\nFigure 3: Natural Emergence of Happiness F(t)')
    
    # Figure 4: Sensitivity Heatmap (This is a simplified heatmap; it can be replaced with actual sensitivity data).
    # 圖 4：敏感性熱圖（這裡用簡單熱圖示意，實際可替換為真實敏感性數據）
    axs[3].set_title('圖 4：k 與 v 的敏感性熱圖 (概念示意)\nFigure 4: Sensitivity Heatmap of k and v (Conceptual)')
    
    # Simple heatmap illustration 簡單示意熱圖
    k_vals = np.linspace(0.1, 1.0, 20)
    v_vals = np.linspace(0.01, 0.1, 20)
    K, V = np.meshgrid(k_vals, v_vals)
    Z = K * V * 100   # Simplified schematic values 簡化示意值
    
    sns.heatmap(Z, ax=axs[3], cmap='viridis', cbar_kws={'label': 'C(100)'})
    axs[3].set_xlabel('k (Pain conversion efficiency 痛苦轉化效率)')
    axs[3].set_ylabel('v (Asymmetry maintenance rate 不對稱性維持速率)')
    
    for ax in axs:
        ax.legend(fontsize=9)
        ax.set_xlabel('時間 t / Time t')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/combined_figures.png', dpi=300, bbox_inches='tight')
    print("The four charts have been successfully generated and saved to the figures/ folder.")
    print("四張圖表已成功生成並儲存至 figures/ 資料夾")

# ====================== 主程式 ======================
if __name__ == "__main__":
    generate_figures()
    print("Simulation complete. Please check the charts in the figures/ folder.")
    print("模擬完成。請檢查 figures/ 資料夾中的圖表。")
