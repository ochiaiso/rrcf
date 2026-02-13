import paho.mqtt.client as mqtt
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from _rrcf import AnomalyDetector, VibrationFeatures
import traceback
import time

# --- 設定 ---
FS = 25000
TOPIC = "vibration/data"
BROKER = "localhost"
UPDATE_INTERVAL_MS = 5000  # グラフ更新間隔 (5秒 = 5000ms)
MAX_PLOT_POINTS = 6000      # グラフに表示する最大データ点数（メモリ制限用）
TIME_MARGIN_RATIO = 0.05    # 時間軸の右側余裕 (5%)
WARMUP_TIME = 30.0          # 慣らし時間 (30秒)
ANOMALY_THRESHOLD = 3.0     # 異常判定の標準偏差倍数

# 表示モード設定
DISPLAY_MODE = "scroll"  # "full" or "scroll"
SCROLL_WINDOW = 60.0     # スクロールモードでの表示時間幅（秒）

# --- 異常検知器の初期化 ---
vf = VibrationFeatures()
detector = AnomalyDetector(
    feature_functions=[vf.calc_rms, vf.calc_spectral_centroid,vf.calc_spectral_centroid],
    shingle_size=10,
    tree_size=100
)

# --- データ保持用バッファ ---
score_history = []
time_history = []
waveform_data_all = []  # 全波形データを保存（2500点ごとのチャンクを連結）
anomaly_flags = []      # 異常フラグ保存用（0.1秒ごと）
current_time = 0
last_update_time = 0
message_count = 0
is_connected = False

def on_connect(client, userdata, flags, rc):
    """MQTTブローカーに接続した時のコールバック"""
    global is_connected
    if rc == 0:
        print("✓ Connected to MQTT Broker successfully!")
        is_connected = True
        # 再接続時にも必ず再サブスクライブ
        client.subscribe(TOPIC)
        print(f"✓ Subscribed to topic: {TOPIC}")
    else:
        print(f"✗ Failed to connect, return code {rc}")
        is_connected = False

def on_disconnect(client, userdata, rc):
    """MQTTブローカーから切断された時のコールバック"""
    global is_connected
    is_connected = False
    if rc != 0:
        print(f"\n⚠ Unexpected disconnection. Return code: {rc}")
        print("⟳ Reconnection will be handled automatically by loop_start()")
    else:
        print("✓ Disconnected normally")

def on_message(client, userdata, msg):
    global current_time, message_count, last_update_time
    
    try:
        payload = json.loads(msg.payload)
        waveform_chunk = np.array(payload["data"])
        
        # スコア計算 (0.1s分のデータから1つのスコア)
        score = detector.get_score(waveform_chunk, fs=FS)
        
        # 生の波形データを保存
        waveform_data_all.extend(waveform_chunk.tolist())
        
        # 異常判定（30秒経過後のみ）
        is_anomaly = False
        if current_time >= WARMUP_TIME and len(score_history) > 0:
            valid_scores = [s for s in score_history if s > 0.0]
            if len(valid_scores) > 0:
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                upper_threshold = mean_score + ANOMALY_THRESHOLD * std_score
                # 上限を超えた場合に異常と判定
                if score > upper_threshold:
                    is_anomaly = True
        
        message_count += 1
        
        # 10メッセージごとに詳細ログ、それ以外は簡易ログ
        anomaly_mark = " ⚠ ANOMALY!" if is_anomaly else ""
        if message_count % 10 == 0:
            print(f"[MSG #{message_count}] Time: {current_time:.1f}s, Score: {score:.4f}, Buffer: {len(score_history)}{anomaly_mark}")
        else:
            print(f"Time: {current_time:.1f}s, Score: {score:.4f}{anomaly_mark}")
        
        # データをバッファに保存
        score_history.append(score)
        time_history.append(current_time)
        anomaly_flags.append(is_anomaly)
        current_time += 0.1  # 送信側が0.1sおきなので
        last_update_time = current_time
        
        # メモリ節約のため古いデータを捨てる
        if len(score_history) > MAX_PLOT_POINTS:
            score_history.pop(0)
            time_history.pop(0)
            anomaly_flags.pop(0)
            # 波形データも古い0.1秒分（2500点）を削除
            waveform_data_all[:2500] = []
            
    except Exception as e:
        print(f"✗ Error in on_message: {e}")
        traceback.print_exc()

# --- グラフ設定（上下2段） ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# 上段：波形データ（RMS）
line1, = ax1.plot([], [], 'b-', lw=1, label='Waveform RMS')
ax1.set_title("Waveform Data (RMS per 0.1s chunk)", fontsize=14, fontweight='bold')
ax1.set_ylabel("RMS Amplitude", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 下段：異常スコア
line2, = ax2.plot([], [], 'r-', lw=2, label='Anomaly Score')
ax2.set_title("Real-time Anomaly Score (25kHz Vibration Analysis)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Time (s)", fontsize=12)
ax2.set_ylabel("Anomaly Score", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

# 統計情報表示用のテキストオブジェクト（下段グラフの左下に配置）
stats_text = ax2.text(0.02, 0.02, '', transform=ax2.transAxes,
                     verticalalignment='bottom', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                     fontsize=10, fontfamily='monospace')

# デバッグ情報表示用（下段グラフの右上に配置）
debug_text = ax2.text(0.98, 0.98, '', transform=ax2.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                     fontsize=9, fontfamily='monospace')

def init_plot():
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 15)
    line1.set_data([], [])
    line2.set_data([], [])
    stats_text.set_text('')
    debug_text.set_text('')
    return line1, line2, stats_text, debug_text

# グラフ更新カウンター
plot_update_count = 0

def update_plot(frame):
    global plot_update_count
    plot_update_count += 1
    
    try:
        if plot_update_count % 10 == 0:
            print(f"\n[PLOT UPDATE #{plot_update_count}] Frame: {frame}, Data points: {len(score_history)}")
        
        if not time_history:
            connection_status = "🟢 Connected" if is_connected else "🔴 Disconnected"
            debug_text.set_text(f'Waiting for data...\n{connection_status}')
            return line1, line2, stats_text, debug_text
        
        # 表示範囲の計算
        if DISPLAY_MODE == "scroll":
            # スクロールモード：最新60秒を表示
            latest_time = time_history[-1]
            x_min = max(0, latest_time - SCROLL_WINDOW)
            x_max = latest_time + SCROLL_WINDOW * 0.02  # 少し余裕を持たせる
        else:
            # フルモード：全データを表示
            latest_time = time_history[-1]
            x_min = 0
            x_max = latest_time * (1 + TIME_MARGIN_RATIO)
            x_max = max(x_max, 10)
        
        # 上段グラフ（生波形）を更新
        ax1.clear()
        ax1.set_ylabel("Amplitude", fontsize=12)
        ax1.set_title("Waveform Data (25kHz Raw Signal)", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        if len(waveform_data_all) > 0:
            waveform_array = np.array(waveform_data_all)
            
            # 異常フラグに基づいて色分け
            # 各チャンクは2500点なので、各0.1秒区間ごとに色を変える
            for i, is_anomaly in enumerate(anomaly_flags):
                start_idx = i * 2500
                end_idx = min(start_idx + 2500, len(waveform_array))
                
                if start_idx < len(waveform_array) and end_idx > start_idx:
                    # この区間の時間軸を作成（開始時刻 + サンプル番号/サンプリング周波数）
                    start_time = i * 0.1  # 各チャンクは0.1秒
                    segment_length = end_idx - start_idx
                    segment_time = start_time + np.arange(segment_length) / FS
                    segment_data = waveform_array[start_idx:end_idx]
                    
                    # 表示範囲内のデータのみプロット
                    if segment_time[-1] >= x_min and segment_time[0] <= x_max:
                        if is_anomaly:
                            ax1.plot(segment_time, segment_data, 'r-', lw=0.5, alpha=0.8)
                        else:
                            ax1.plot(segment_time, segment_data, 'b-', lw=0.5, alpha=0.6)
            
            # 凡例用のダミープロット
            ax1.plot([], [], 'b-', lw=2, label='Normal', alpha=0.6)
            ax1.plot([], [], 'r-', lw=2, label='Anomaly', alpha=0.8)
            ax1.legend()
        
        # 下段グラフ（異常スコア）を更新
        ax2.clear()
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("Anomaly Score", fontsize=12)
        ax2.set_title("Real-time Anomaly Score (25kHz Vibration Analysis)", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # スコアも色分けして表示（表示範囲内のみ）
        for i in range(len(time_history)):
            if i > 0 and time_history[i] >= x_min and time_history[i-1] <= x_max:
                # 線分を描画
                if anomaly_flags[i] or anomaly_flags[i-1]:
                    ax2.plot([time_history[i-1], time_history[i]], 
                            [score_history[i-1], score_history[i]], 
                            'r-', lw=2, alpha=0.8)
                else:
                    ax2.plot([time_history[i-1], time_history[i]], 
                            [score_history[i-1], score_history[i]], 
                            'b-', lw=2, alpha=0.6)
        
        # ポイントマーカーを追加（表示範囲内のみ）
        normal_times = [time_history[i] for i in range(len(time_history)) 
                       if not anomaly_flags[i] and x_min <= time_history[i] <= x_max]
        normal_scores = [score_history[i] for i in range(len(score_history)) 
                        if not anomaly_flags[i] and x_min <= time_history[i] <= x_max]
        anomaly_times = [time_history[i] for i in range(len(time_history)) 
                        if anomaly_flags[i] and x_min <= time_history[i] <= x_max]
        anomaly_scores = [score_history[i] for i in range(len(score_history)) 
                         if anomaly_flags[i] and x_min <= time_history[i] <= x_max]
        
        if normal_times:
            ax2.plot(normal_times, normal_scores, 'bo', markersize=3, label='Normal', alpha=0.6)
        if anomaly_times:
            ax2.plot(anomaly_times, anomaly_scores, 'ro', markersize=5, label='Anomaly', alpha=0.8)
        
        ax2.legend()
        
        # 横軸の範囲を設定
        ax1.set_xlim(x_min, x_max)
        ax2.set_xlim(x_min, x_max)
        
        # 上段の縦軸の範囲を自動調整（表示範囲内のデータのみ考慮）
        if len(waveform_data_all) > 0:
            waveform_array = np.array(waveform_data_all)
            # 表示範囲内のデータを抽出
            visible_indices = []
            for i in range(len(anomaly_flags)):
                start_time = i * 0.1
                end_time = start_time + 0.1
                if start_time <= x_max and end_time >= x_min:
                    start_idx = i * 2500
                    end_idx = min(start_idx + 2500, len(waveform_array))
                    visible_indices.extend(range(start_idx, end_idx))
            
            if visible_indices:
                visible_data = waveform_array[visible_indices]
                max_w = np.max(visible_data)
                min_w = np.min(visible_data)
                margin = (max_w - min_w) * 0.1
                ax1.set_ylim(min_w - margin, max_w + margin)
        
        # 下段の縦軸の範囲を自動調整（表示範囲内のデータのみ考慮）
        visible_scores = [score_history[i] for i in range(len(score_history)) 
                         if x_min <= time_history[i] <= x_max]
        if visible_scores:
            max_s = max(visible_scores)
            min_s = min(visible_scores)
            if max_s > 0:
                ax2.set_ylim(min(0, min_s * 0.9), max_s * 1.2)
            else:
                ax2.set_ylim(0, 15)
        
        # 統計情報の計算と表示更新（全データに基づく）
        if len(score_history) > 0:
            valid_scores = [s for s in score_history if s > 0.0]
            
            if len(valid_scores) > 0:
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                min_score = np.min(valid_scores)
                max_score = np.max(valid_scores)
                upper_threshold = mean_score + ANOMALY_THRESHOLD * std_score
                anomaly_count = sum(anomaly_flags)
                
                stats_info = f'Mean:       {mean_score:.4f}\nStd:        {std_score:.4f}\nMin:        {min_score:.4f}\nMax:        {max_score:.4f}\nThreshold:  {upper_threshold:.4f}\nAnomalies:  {anomaly_count}\nN:          {len(valid_scores)}'
            else:
                stats_info = 'Waiting for data...'
            
            stats_text.set_text(stats_info)
            stats_text.set_position((0.02, 0.02))
            stats_text.set_transform(ax2.transAxes)
        
        # デバッグ情報の更新（接続状態を含む）
        connection_status = "🟢" if is_connected else "🔴"
        warmup_status = "⏱ Warmup" if current_time < WARMUP_TIME else "✓ Active"
        mode_info = f"Mode: {DISPLAY_MODE.upper()}"
        debug_info = f'{connection_status} Update: #{plot_update_count}\nMsgs:   {message_count}\nLast:   {last_update_time:.1f}s\nBuffer: {len(score_history)}\nWave:   {len(waveform_data_all)} pts\n{warmup_status}\n{mode_info}'
        debug_text.set_text(debug_info)
        debug_text.set_position((0.98, 0.98))
        debug_text.set_transform(ax2.transAxes)
        
        # 軸の範囲を変更したので再描画
        fig.canvas.draw()
        
        return line1, line2, stats_text, debug_text
        
    except Exception as e:
        print(f"✗ Error in update_plot: {e}")
        traceback.print_exc()
        return line1, line2, stats_text, debug_text

# --- メイン処理 ---
def start_receiver():
    # Client IDを設定して、再接続時の識別を容易に
    client = mqtt.Client(client_id="vibration_monitor", clean_session=True)
    
    # 自動再接続の設定
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    
    try:
        print(f"⟳ Connecting to MQTT broker at {BROKER}:1883...")
        client.connect(BROKER, 1883, 60)
        
        # MQTTの受信ループを別スレッドで開始（自動再接続が有効）
        client.loop_start()
        print("✓ MQTT loop started (auto-reconnect enabled)")
        
        # 接続待機
        timeout = 10
        start_time = time.time()
        while not is_connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not is_connected:
            print("⚠ Warning: Initial connection timeout, but continuing...")
        
        # グラフの表示更新
        print(f"\n{'='*50}")
        print(f"Configuration:")
        print(f"  Graph update interval: {UPDATE_INTERVAL_MS/1000}s")
        print(f"  Max plot points: {MAX_PLOT_POINTS} ({MAX_PLOT_POINTS * 0.1}s)")
        if DISPLAY_MODE == "scroll":
            print(f"  Display mode: SCROLL (last {SCROLL_WINDOW}s)")
        else:
            print(f"  Display mode: FULL (0 to current_time × {1 + TIME_MARGIN_RATIO})")
        print(f"  Warmup time: {WARMUP_TIME}s")
        print(f"  Anomaly threshold: Mean + {ANOMALY_THRESHOLD}σ")
        print(f"{'='*50}\n")
        
        ani = FuncAnimation(fig, update_plot, init_func=init_plot, 
                            interval=UPDATE_INTERVAL_MS, blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
        
    except KeyboardInterrupt:
        print("\n⟳ Shutting down gracefully...")
    except Exception as e:
        print(f"✗ Error in start_receiver: {e}")
        traceback.print_exc()
    finally:
        client.loop_stop()
        client.disconnect()
        print("✓ Disconnected from MQTT broker")

if __name__ == "__main__":
    start_receiver()