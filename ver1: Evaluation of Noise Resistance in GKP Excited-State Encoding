# 1. Strawberry Fields のインストール
!pip install strawberryfields

# 2. SciPy 互換性パッチ (Python 3.13 / SciPy 1.14+ 対応)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    from scipy.integrate import simpson
    scipy.integrate.simps = simpson

import scipy.special
if not hasattr(scipy.special, 'factorial'):
    from scipy.special import factorial
    scipy.special.factorial = factorial

# 3. モジュールのインポート
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import Sgate, Dgate, Kgate, Rgate, LossChannel

# 4. ハイパーパラメータと実験設定
CUTOFF = 15                 # Fock空間のカットオフ次元
LOSS_RATES = [0.0, 0.05, 0.10, 0.20]  # 光子損失率 (0%, 5%, 10%, 20%)
X_VEC = np.linspace(-4, 4, 100)        # Wigner関数描画用の位相空間軸
P_VEC = np.linspace(-4, 4, 100)

def gkp_approx_state(q, x_val, sq_r=0.8):
    Sgate(sq_r, 0) | q
    Dgate(x_val, 0) | q

def squeezed_state_encoding(q, x_val, sq_r=0.4):
    Sgate(sq_r, 0) | q
    Dgate(x_val, 0) | q

def apply_ansatz(q, params):
    r, phi, d_mag, k_val = params
    Rgate(phi) | q
    Sgate(r, 0) | q
    Dgate(d_mag, 0) | q
    Kgate(k_val) | q

def run_cv_vqc_simulation(x_data, ansatz_params, loss_rate, mode_type="gkp"):
    prog = sf.Program(1)
    with prog.context as q:
        if mode_type == "gkp":
            gkp_approx_state(q[0], x_data)
        else:
            squeezed_state_encoding(q[0], x_data)

        if loss_rate > 0.0:
            LossChannel(1.0 - loss_rate) | q[0]

        apply_ansatz(q[0], ansatz_params)

    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": CUTOFF})
    results = eng.run(prog)
    return results.state

# 5. 実行と Wigner 関数の視覚化
test_params = [0.3, np.pi / 4, 0.5, 0.1]
x_input = 1.2

fig, axes = plt.subplots(len(LOSS_RATES), 2, figsize=(10, 12))

for i, loss in enumerate(LOSS_RATES):
    # GKP 励起型エンコーディングの実行
    state_gkp = run_cv_vqc_simulation(x_input, test_params, loss, mode_type="gkp")
    W_gkp = state_gkp.wigner(0, X_VEC, P_VEC)  # 単一変数で受け取り

    # 圧縮状態ベースラインの実行
    state_sq = run_cv_vqc_simulation(x_input, test_params, loss, mode_type="squeezed")
    W_sq = state_sq.wigner(0, X_VEC, P_VEC)   # 単一変数で受け取り

    # プロット (GKP)
    axes[i, 0].contourf(X_VEC, P_VEC, W_gkp, 100, cmap="RdBu_r")
    axes[i, 0].set_title(f"GKP-like (Loss: {int(loss*100)}%)")
    axes[i, 0].set_xlabel("x")
    axes[i, 0].set_ylabel("p")

    # プロット (Squeezed)
    axes[i, 1].contourf(X_VEC, P_VEC, W_sq, 100, cmap="RdBu_r")
    axes[i, 1].set_title(f"Squeezed Baseline (Loss: {int(loss*100)}%)")
    axes[i, 1].set_xlabel("x")
    axes[i, 1].set_ylabel("p")

plt.tight_layout()
plt.show()
