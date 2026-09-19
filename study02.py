# ---------------------------------------------------------
# ライブラリのインポートおよび互換性補正
# ---------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- SciPy 互換性補正 ---
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import strawberryfields as sf
from strawberryfields.ops import Sgate, Xgate, Zgate, LossChannel, MeasureHomodyne, Rgate, BSgate, Thermal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 1. 一般CVノイズチャネルの構築と量子状態シミュレーション
# =========================================================

def simulate_cv_channel_and_generate_dataset(
    num_samples=2000,
    squeezing_db=10.0,
    loss_rate=0.08,
    thermal_nbar=0.05,
    phase_damping=0.02,
    cutoff_dim=15
):
    print(f"Generating CV Noise Dataset ({num_samples} samples)...")
    print(f" - Loss Rate: {loss_rate}, Thermal nbar: {thermal_nbar}, Phase Damping: {phase_damping}")

    r = squeezing_db * np.log(10) / 20.0
    transmissivity = 1.0 - loss_rate
    theta = np.arccos(np.sqrt(transmissivity))  # ビームスプリッターの回転角

    q_syndromes = []
    p_syndromes = []
    initial_logical_states = []

    for i in range(num_samples):
        logical_bit = np.random.choice([0, 1])
        initial_logical_states.append(logical_bit)

        # ------------------ q軸測定サンプリング ------------------
        eng_q = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff_dim})
        # 2モード用意 (q[0]: メイン状態, q[1]: 熱環境モード)
        prog_q = sf.Program(2)
        with prog_q.context as q:
            # 1. 状態準備
            Sgate(r) | q[0]
            if logical_bit == 1:
                Xgate(np.sqrt(np.pi)) | q[0]

            # 2. ThermalLossChannel の物理等価実装
            if thermal_nbar > 0:
                sf.ops.Thermal(thermal_nbar) | q[1]
            BSgate(theta, 0.0) | (q[0], q[1])  # ビームスプリッターによる光子損失と熱混入

            # 3. 位相拡散ノイズ
            if phase_damping > 0:
                phase_noise = np.random.normal(0, phase_damping)
                Rgate(phase_noise) | q[0]

            # 4. ホモダイン測定
            MeasureHomodyne(0.0) | q[0]

        results_q = eng_q.run(prog_q)
        q_res = results_q.samples[0][0]

        # ------------------ p軸測定サンプリング ------------------
        eng_p = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff_dim})
        prog_p = sf.Program(2)
        with prog_p.context as q:
            # 1. 状態準備
            Sgate(r) | q[0]
            if logical_bit == 1:
                Xgate(np.sqrt(np.pi)) | q[0]

            # 2. ThermalLossChannel の物理等価実装
            if thermal_nbar > 0:
                sf.ops.Thermal(thermal_nbar) | q[1]
            BSgate(theta, 0.0) | (q[0], q[1])

            # 3. 位相拡散ノイズ
            if phase_damping > 0:
                phase_noise = np.random.normal(0, phase_damping)
                Rgate(phase_noise) | q[0]

            # 4. ホモダイン測定
            MeasureHomodyne(np.pi / 2) | q[0]

        results_p = eng_p.run(prog_p)
        p_res = results_p.samples[0][0]

        q_syndromes.append(q_res)
        p_syndromes.append(p_res)

    return (
        np.array(q_syndromes, dtype=np.float32),
        np.array(p_syndromes, dtype=np.float32),
        np.array(initial_logical_states, dtype=np.int64)
    )
# =========================================================
# 2. CVシンドロームから論理マルコフノイズモデル（Pauli Error Matrix）への射影
# =========================================================
def cv_to_logical_pauli_channel(q_measured, p_measured, initial_states):
    """
    連続測定値 (q, p) から GKP 格子モジュロ演算を用いて
    論理パウリエラー (I, X, Y, Z) の推論プロセスのチャネル変換行列（マルコフ核）を計算する。
    """
    sqrt_pi = np.sqrt(np.pi)

    # モジュロ演算によるシンドローム抽出: [-sqrt(pi)/2, sqrt(pi)/2] への折り返し
    delta_q = np.mod(q_measured + sqrt_pi / 2.0, sqrt_pi) - sqrt_pi / 2.0
    delta_p = np.mod(p_measured + sqrt_pi / 2.0, sqrt_pi) - sqrt_pi / 2.0

    # 単純格子デコードによる推定論理シフト (最近傍格子の判定)
    nearest_q_lattice = np.round((q_measured - delta_q) / sqrt_pi)
    nearest_p_lattice = np.round((p_measured - delta_p) / sqrt_pi)

    # 奇数格子の場合は論理反転 (Xエラー / Zエラー) が発生したと判定
    x_error = (np.abs(nearest_q_lattice) % 2 == 1).astype(int)
    z_error = (np.abs(nearest_p_lattice) % 2 == 1).astype(int)

    # 論理パウリエラー分類 (0: I, 1: X, 2: Z, 3: Y)
    pauli_events = np.zeros(len(q_measured), dtype=int)
    pauli_events[(x_error == 1) & (z_error == 0)] = 1 # X
    pauli_events[(x_error == 0) & (z_error == 1)] = 2 # Z
    pauli_events[(x_error == 1) & (z_error == 1)] = 3 # Y

    # 確率分布 (マルコフノイズ遷移確率) の算出
    total = len(q_measured)
    p_I = np.sum(pauli_events == 0) / total
    p_X = np.sum(pauli_events == 1) / total
    p_Z = np.sum(pauli_events == 2) / total
    p_Y = np.sum(pauli_events == 3) / total

    pauli_probs = {'I': p_I, 'X': p_X, 'Z': p_Z, 'Y': p_Y}
    return pauli_probs, pauli_events

# =========================================================
# 3. 特徴量抽出とMLデコーダ (最尤推定ニューラルネットワーク)
# =========================================================
class CVLogicalDecoderMLP(nn.Module):
    """
    測定値 (q, p) および非線形モジュロ特徴量から、
    論理パウリエラー (I, X, Z, Y) の4クラス分類を行うMLP。
    """
    def __init__(self):
        super(CVLogicalDecoderMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 4) # 出力: [Prob(I), Prob(X), Prob(Z), Prob(Y)]
        )

    def forward(self, x):
        return self.net(x)

def extract_cv_features(q_vals, p_vals):
    sqrt_pi = np.sqrt(np.pi)

    q_mod = (np.mod(q_vals + sqrt_pi / 2, sqrt_pi) - sqrt_pi / 2).reshape(-1, 1)
    p_mod = (np.mod(p_vals + sqrt_pi / 2, sqrt_pi) - sqrt_pi / 2).reshape(-1, 1)

    q_sin = np.sin(2 * np.pi * q_vals / sqrt_pi).reshape(-1, 1)
    q_cos = np.cos(2 * np.pi * q_vals / sqrt_pi).reshape(-1, 1)
    p_sin = np.sin(2 * np.pi * p_vals / sqrt_pi).reshape(-1, 1)
    p_cos = np.cos(2 * np.pi * p_vals / sqrt_pi).reshape(-1, 1)

    return np.hstack([q_mod, p_mod, q_sin, q_cos, p_sin, p_cos])

# =========================================================
# 4. メイン評価パイプライン
# =========================================================
def main():
    # --- A. 一般CVノイズ環境でのデータ生成 ---
    q_train, p_train, states_train = simulate_cv_channel_and_generate_dataset(num_samples=3000)
    q_test, p_test, states_test = simulate_cv_channel_and_generate_dataset(num_samples=1000)

    # --- B. 物理ノイズから論理マルコフノイズモデル (Pauli Channel) への変換 ---
    raw_pauli_probs, raw_pauli_events = cv_to_logical_pauli_channel(q_test, p_test, states_test)
    _, train_pauli_events = cv_to_logical_pauli_channel(q_train, p_train, states_train)

    print("\n================ 抽出された論理マルコフノイズモデル (Pauli Channel) ================")
    print(f"P(Identity / No-Error) : {raw_pauli_probs['I'] * 100:.2f}%")
    print(f"P(Pauli X Error)       : {raw_pauli_probs['X'] * 100:.2f}%")
    print(f"P(Pauli Z Error)       : {raw_pauli_probs['Z'] * 100:.2f}%")
    print(f"P(Pauli Y Error)       : {raw_pauli_probs['Y'] * 100:.2f}%")
    print("====================================================================================\n")

    # --- C. MLデコーダの学習 ---
    X_train = torch.tensor(extract_cv_features(q_train, p_train), dtype=torch.float32)
    y_train = torch.tensor(train_pauli_events, dtype=torch.int64)
    X_test = torch.tensor(extract_cv_features(q_test, p_test), dtype=torch.float32)
    y_test = torch.tensor(raw_pauli_events, dtype=torch.int64)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    model = CVLogicalDecoderMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("--- Training ML Decoder on General CV Noise ---")
    model.train()
    for epoch in range(20):
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/20 | Loss: {total_loss / len(X_train):.4f}")

    # --- D. 残存論理エラー率の評価 ---
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # パウリエラーの誤判定率（残存論理エラー率）
    uncorrected_logical_error = np.mean(raw_pauli_events != 0)

    # MLデコーダが正しくパウリエラーを同定できず、間違った補正を適用してしまった割合
    residual_logical_error = np.mean(preds != y_test.numpy())

    print("\n================ 評価結果 (残存論理エラー率) ================")
    print(f"物理ノイズ起因の論理エラー率 (Raw Logical Error Rate) : {uncorrected_logical_error * 100:.2f}%")
    print(f"MLデコーダ適用後の残存論理エラー率 (Residual Logical ER): {residual_logical_error * 100:.2f}%")
    print("===============================================================")

if __name__ == "__main__":
    main()
