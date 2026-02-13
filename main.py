from vibe_analyzer import VibrationFeatures
from rrcf._rrcf import AnomalyDetector

# 1. 使用する特徴量関数をリストアップ（ここを増減させるだけでOK）
vf = VibrationFeatures()
selected_features = [
    vf.calc_rms,
    vf.calc_kurtosis,
    vf.calc_spectral_centroid
]

# 2. 検知器の初期化
# shingle_sizeを大きくすると、より「長期的な変化（定常状態の変化）」に敏感になります
detector = AnomalyDetector(
    feature_functions=selected_features, 
    shingle_size=12, 
    num_trees=50
)

# 3. リアルタイムループ（模擬）
# while True: 
#    data = get_sensor_data() 
#    score = detector.get_score(data, fs=10000)
#    if score > threshold: 
#        print("Alert!")