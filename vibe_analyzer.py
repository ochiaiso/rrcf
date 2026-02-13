import numpy as np
from scipy import stats

class VibrationFeatures:
    """振動データの各種特徴量を計算するライブラリ"""

    @staticmethod
    def calc_rms(data):
        """実効値 (Root Mean Square) の計算"""
        # RMSは時間領域のエネルギーなので窓関数は不要
        return np.sqrt(np.mean(np.square(data)))

    @staticmethod
    def calc_kurtosis(data):
        """尖度 (Kurtosis) の計算"""
        # 統計量も生データ（またはトレンド除去後）に対して行う
        return stats.kurtosis(data)

    @staticmethod
    def calc_spectral_centroid(data, fs):
        """
        窓関数を適用した重心周波数 (Spectral Centroid) の計算
        """
        # 1. 窓関数（ハニング窓）の適用
        # データの両端をスムーズに0に落とし、周波数リーケージを抑制する
        window = np.hanning(len(data))
        windowed_data = data * window
        
        # 2. FFT実行 (Real FFT)
        spectrum = np.abs(np.fft.rfft(windowed_data))
        frequencies = np.fft.rfftfreq(len(data), d=1/fs)
        
        # 3. 重心計算: Σ(周波数 * 強度) / Σ(強度)
        sum_spectrum = np.sum(spectrum)
        if sum_spectrum == 0:
            return 0
            
        return np.sum(frequencies * spectrum) / sum_spectrum