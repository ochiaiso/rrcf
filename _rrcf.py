import numpy as np
import rrcf
from vibe_analyzer import VibrationFeatures

class AnomalyDetector:
    """RRCFを用いた振動異常検知ライブラリ"""

    def __init__(self, feature_functions, shingle_size=10, num_trees=200, tree_size=1024):
        """
        Args:
            feature_functions: 使用する特徴量関数のリスト (例: [vf.calc_rms, vf.calc_kurtosis])
            shingle_size: 過去何回分の特徴量を束ねるか
            num_trees: RRCFの木の数
            tree_size: 木に保持する最大データ点数
        """
        self.feature_functions = feature_functions
        self.shingle_size = shingle_size
        self.num_features = len(feature_functions)
        
        # RRCFの初期化
        self.forest = [rrcf.RCTree() for _ in range(num_trees)]
        self.tree_size = tree_size
        
        # Shingling用のバッファ（過去の特徴量ベクトルを保持）
        self.shingle_buffer = []
        self.total_points = 0

        # _rrcf.py の AnomalyDetector.__init__ に追加
        self.mean = None
        self.std = None
        self.alpha = 0.01 # 更新重み（適宜調整）

    def get_score(self, time_series_data, fs):
        # 1. 特徴量抽出
        features = []
        for func in self.feature_functions:
            # --- 修正ポイント ---
            # 関数名を確認し、fsが必要な関数（スペクトル系）にはfsを渡す
            if func.__name__ in ['calc_spectral_centroid', 'calc_centroid']:
                val = func(time_series_data, fs=fs)
            else:
                val = func(time_series_data)
            features.append(val)

        # get_score メソッド内、features計算直後に追加
        features = np.array(features)
        if self.mean is None:
            self.mean = features
            self.std = np.ones_like(features)
        else:
            # 逐次更新（指数移動平均的なアプローチ）
            self.mean = (1 - self.alpha) * self.mean + self.alpha * features
            self.std = (1 - self.alpha) * self.std + self.alpha * np.abs(features - self.mean)

        # 標準化 (z-score)
        normalized_features = (features - self.mean) / (self.std + 1e-9)
        self.shingle_buffer.append(normalized_features)

        # 2. Shingle（過去の履歴と結合）に変換
        #self.shingle_buffer.append(features)
        
        # バッファサイズをshingle_sizeに保つ（古いデータを削除）
        if len(self.shingle_buffer) > self.shingle_size:
            self.shingle_buffer.pop(0)
        
        # 履歴が貯まるまでは0を返す
        if len(self.shingle_buffer) < self.shingle_size:
            return 0.0
            
        shingle = np.array(list(self.shingle_buffer)).flatten()
        
        # --- 重複エラー対策 & スコア計算 ---
        # わずかなノイズを加えて、RRCFが「同じ点だ！」と判定するのを防ぐ
        shingle += np.random.normal(0, 1e-10, shingle.shape)
        
        scores = []
        for tree in self.forest:
            # 100点（tree_size）貯まったら古い点から消していく
            if len(tree.leaves) >= self.tree_size:
                # 0番（一番古い点）を削除
                # tree.leavesのキーから最小のものを消すのが確実
                oldest_idx = min(tree.leaves.keys())
                tree.forget_point(oldest_idx)
            
            # 新しい点を挿入（インデックスを重複させないために今のカウントを使用）
            tree.insert_point(shingle, index=self.total_points)
            
            # 異常度の計算 (CoDisp)
            scores.append(tree.codisp(self.total_points))
        
        self.total_points += 1
        return np.mean(scores)