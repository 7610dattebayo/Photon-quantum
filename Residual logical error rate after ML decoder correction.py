# ---------------------------------------------------------
# ライブラリのインポートおよび互換性補正
# ---------------------------------------------------------
import numpy as np                          # 数値計算用ライブラリ
import torch                                # PyTorch本体（深層学習フレームワーク）
import torch.nn as nn                       # ニューラルネットワーク層を構築するためのモジュール
import torch.optim as optim                 # 最適化アルゴリズム（AdamWなど）を提供するモジュール
from torch.utils.data import DataLoader, TensorDataset # データをミニバッチに分割・管理するツール

# --- SciPy 互換性補正コード（Strawberry Fields のインポート前に実行） ---
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson
# ------------------------------------------------------------------

import strawberryfields as sf               # 光量子シミュレータ Strawberry Fields
from strawberryfields.ops import LossChannel, MeasureHomodyne, Sgate, Rgate, Xgate
                                            # 量子ゲートや測定、チャネル操作の定義

# GPU（CUDA）が使用可能であればGPUを、利用できなければCPUを設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Strawberry Fields によるシミュレーション・データ生成
# ==========================================
def generate_gkp_dataset(num_samples=5000, squeezing_db=10.0, loss_rate=0.1):
    """
    GKP状態に対してノイズを印加し、ホモダイン測定データ（q）と
    印加した真のエラー量（delta_q）のデータセットを生成する関数。
    """
    print(f"Generating {num_samples} samples (Squeezing: {squeezing_db}dB, Loss: {loss_rate})...")

    # 圧縮パラメータ r をデシベル(dB)表記から計算（r = dB * ln(10) / 20）
    r = squeezing_db * np.log(10) / 20.0

    # 光子損失チャネルの透過率 T を計算（透過率 = 1 - 損失率）
    transmissivity = 1.0 - loss_rate

    # 生成データを格納するための空リストを用意
    q_measurements = [] # 測定された位置 q のデータ
    true_shifts = []    # 正解となるノイズ（エラーシフト量）データ

    # sf.Engine: 量子プログラムを実行するためのエンジンを初期化
    # backend="fock": フォック状態（光子数）表現によるバックエンドを使用
    # cutoff_dim=15: 計算対象とする最大光子数を15に制限（次元削減）
    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 15})

    # 指定されたサンプル数ぶん量子シミュレーションを繰り返す
    for i in range(num_samples):
        # 1量子モード（1つの光チャネル）を用いる量子プログラムを定義
        prog = sf.Program(1)

        # 人工的な位置エラー（変位ノイズ）を [-sqrt(pi)/2, sqrt(pi)/2] の一様分布でランダム生成
        delta_q = np.random.uniform(-np.sqrt(np.pi)/2, np.sqrt(np.pi)/2)

        # 量子プログラムにゲート操作を追加する文脈（コンテキスト）を開く
        with prog.context as q:
            # 1. 圧縮ゲート (Sgate): 理想的なGKP状態の代用として位置圧縮状態を作成
            Sgate(r) | q[0]

            # 2. シフト操作 (Xgate): 変位ノイズ delta_q を位置方向（q軸）に加える（引数は1つのみ）
            Xgate(delta_q) | q[0]

            # 3. 損失チャネル (LossChannel): 環境からの光子減衰（光子損失）をシミュレート
            LossChannel(transmissivity) | q[0]

            # 4. ホモダイン測定 (MeasureHomodyne): 0.0ラジアン（q軸/位置方向）の測定を行う
            MeasureHomodyne(0.0) | q[0]

        # 作成したプログラムをエンジンで実行
        results = eng.run(prog)

        # 測定結果（qの値）を取得（最初のモードの最初のサンプル）
        measured_q = results.samples[0][0]

        # 取得した測定値と正解ノイズをリストに追加
        q_measurements.append(measured_q)
        true_shifts.append(delta_q)

    # NumPy配列に変換して（データ型をfloat32に固定）返却
    return np.array(q_measurements, dtype=np.float32), np.array(true_shifts, dtype=np.float32)


# ==========================================
# 2. 特徴量エンジニアリング (Feature Engineering)
# ==========================================
def preprocess_features(q_vals):
    """
    測定値 q から、GKPコードの持つ格子周期性 sqrt(pi) を利用した特徴量を作成する関数。
    """
    # GKPコードの基本周期 T = sqrt(pi)
    period = np.sqrt(np.pi)

    # 1. 生の測定値 q（形状を [N, 1] の2次元列ベクトルに整形）
    q_raw = q_vals.reshape(-1, 1)

    # 2. 格子中心からのオフセット（モジュロ演算で周期内に折り返し [-period/2, period/2] に収める）
    q_mod = np.mod(q_vals + period / 2, period) - period / 2
    q_mod = q_mod.reshape(-1, 1)

    # 3. 周期性を表現するための三角関数特徴量 (sin と cos)
    sin_feat = np.sin(2 * np.pi * q_vals / period).reshape(-1, 1)
    cos_feat = np.cos(2 * np.pi * q_vals / period).reshape(-1, 1)

    # 4つの特徴量を横方向（列方向）に連結し、形状 [N, 4] の行列を作成
    X = np.hstack([q_raw, q_mod, sin_feat, cos_feat])
    return X


# ==========================================
# 3. 機械学習デコーダ（PyTorch）の設計
# ==========================================
class GKPDecoderMLP(nn.Module):
    """
    特徴量 (4次元) からエラーシフト量 delta_q (1次元) を予測する多層パーセプトロン (MLP)。
    """
    def __init__(self, input_dim=4):
        super(GKPDecoderMLP, self).__init__()
        # 順伝播のネットワーク構造を定義
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), # 全結合層: 入力4次元 -> 隠れ層64次元
            nn.GELU(),                # 活性化関数: GELU (Gaussian Error Linear Unit)
            nn.Linear(64, 64),        # 全結合層: 隠れ層64次元 -> 隠れ層64次元
            nn.GELU(),                # 活性化関数: GELU
            nn.Linear(64, 32),        # 全結合層: 隠れ層64次元 -> 隠れ層32次元
            nn.GELU(),                # 活性化関数: GELU
            nn.Linear(32, 1)          # 出力層: 隠れ層32次元 -> 予測値1次元 (delta_q)
        )

    def forward(self, x):
        # データをネットワークに通して出力を返す関数
        return self.net(x)

class LogicalErrorAwareLoss(nn.Module):
    """
    通常の2乗誤差(MSE)に加えて、論理境界を超えた大きな予測誤差に強力なペナルティを与えるカスタム損失関数。
    """
    def __init__(self, threshold=np.sqrt(np.pi)/2):
        super(LogicalErrorAwareLoss, self).__init__()
        self.threshold = threshold  # 論理反転が発生する境界線 sqrt(pi)/2
        self.mse = nn.MSELoss()     # 基本となる平均二乗誤差 (MSE)

    def forward(self, y_pred, y_true):
        # 1. 基本となる予測値と正解値の2乗誤差
        base_loss = self.mse(y_pred, y_true)

        # 2. 予測値と正解値の絶対誤差（残存エラー）を計算
        residual_error = torch.abs(y_pred - y_true)

        # 3. エラーが閾値を超えた分（超越分）のみを取り出し、2乗して平均をとる（ペナルティ項）
        penalty = torch.mean(torch.relu(residual_error - self.threshold) ** 2)

        # 基本誤差 ＋ 5倍のペナルティ を最終的な損失（Loss）として返す
        return base_loss + 5.0 * penalty


# ==========================================
# 4. 学習および評価ループ
# ==========================================
def main():
    # --- A. データセット生成 ---
    # 訓練用データ 4,000 件を作成（圧縮度 10dB, 損失率 10%）
    q_train_raw, y_train_raw = generate_gkp_dataset(num_samples=400, squeezing_db=10.0, loss_rate=0.1)
    # テスト用データ 1,000 件を作成（同じ条件）
    q_test_raw, y_test_raw = generate_gkp_dataset(num_samples=100, squeezing_db=10.0, loss_rate=0.1)

    # --- B. 特徴量変換とPyTorchテンソル化 ---
    # 特徴量抽出関数を適用してPyTorchテンソル（float32型）に変換
    X_train = torch.tensor(preprocess_features(q_train_raw), dtype=torch.float32)
    y_train = torch.tensor(y_train_raw.reshape(-1, 1), dtype=torch.float32)
    X_test = torch.tensor(preprocess_features(q_test_raw), dtype=torch.float32)
    y_test = torch.tensor(y_test_raw.reshape(-1, 1), dtype=torch.float32)

    # 特徴量(X)とラベル(y)をバインドしたデータセットを作成
    train_dataset = TensorDataset(X_train, y_train)
    # バッチサイズ 64、シャッフルありでデータを給仕するデータローダーを設定
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # --- C. モデル・最適化関数の初期化 ---
    model = GKPDecoderMLP(input_dim=4).to(device)  # モデルを生成し指定デバイス(CPU/GPU)に転送
    criterion = LogicalErrorAwareLoss()           # 上で定義したカスタム損失関数を使用
    # 重み減衰（Weight Decay）付きのAdam最適化アルゴリズムを設定 (学習率 0.001)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # --- D. モデルの学習ループ ---
    epochs = 30  # エポック数（全体データを学習させる回数）
    print("\n--- Training ML Decoder ---")
    model.train() # モデルを訓練モードに設定

    for epoch in range(epochs):
        total_loss = 0.0 # 1エポック分の累計損失値

        # ミニバッチ単位でデータを読み込んで学習
        for batch_x, batch_y in train_loader:
            # データをデバイス(CPU/GPU)に転送
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()         # 勾配の初期化（前回の勾配をリセット）
            outputs = model(batch_x)      # 順伝播（モデルによるエラー量の予測）
            loss = criterion(outputs, batch_y) # 損失（Loss）の計算
            loss.backward()               # 逆伝播（勾配の計算）
            optimizer.step()              # パラメータの更新

            # 損失値にバッチサイズを掛け合わせて累積
            total_loss += loss.item() * batch_x.size(0)

        # 5エポックごとに平均損失を表示
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss / len(train_dataset):.6f}")

    # --- E. 残存論理エラー率の算出・評価 ---
    model.eval() # モデルを評価モードに設定（ドロップアウト等の挙動固定）
    with torch.no_grad(): # 勾配計算をオフにしてメモリ節約
        # テストデータをモデルに入力し、結果を1次元NumPy配列として取得
        preds = model(X_test.to(device)).cpu().numpy().flatten()

    y_test_np = y_test.numpy().flatten() # テスト用の正解ノイズ値
    period = np.sqrt(np.pi)               # GKP格子の周期 sqrt(pi)
    boundary = period / 2.0               # 論理境界 sqrt(pi)/2 （これを超えると誤った判定になる）

    # 補正無しの状態：ノイズそのものが境界値を超えているかどうかで判定
    uncorrected_errors = np.abs(y_test_np) > boundary
    uncorrected_error_rate = np.mean(uncorrected_errors) # エラー率（割合）を計算

    # MLデコーダ補正後：正解ノイズからモデルの予測値を引き算（残存ノイズ）
    residual_shifts = y_test_np - preds
    # 残ったノイズが境界値を超えているか判定（＝論理エラーの発生）
    corrected_errors = np.abs(residual_shifts) > boundary
    corrected_error_rate = np.mean(corrected_errors) # 補正後の論理エラー率

    # 評価結果の表示
    print("\n================ 評価結果 ================")
    print(f"補正前の論理エラー率 (Raw Error Rate)       : {uncorrected_error_rate * 100:.2f}%")
    print(f"MLデコーダ補正後の残存論理エラー率 (Logical ER) : {corrected_error_rate * 100:.2f}%")
    print("==========================================")

# このスクリプトが直接実行された場合に main() を呼び出す処理
if __name__ == "__main__":
    main()
