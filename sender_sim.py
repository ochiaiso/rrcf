import pandas as pd
import numpy as np
import paho.mqtt.client as mqtt
import time
import json

# 設定
CSV_FILE = "../new_waveform_25khz.csv"
TOPIC = "vibration/data"
BROKER = "localhost" # ラズパイとPCが別ならPCのIPアドレス
WINDOW_SIZE = 2500  # 0.1秒分 (25kHz)

def start_sender():
    # データの読み込み（メモリ節約のためchunk読み推奨ですが、まずは全読み）
    print("Loading CSV...")
    df = pd.read_csv(CSV_FILE, header=None)
    vibration_data = df.iloc[:, 1].values # 2列目が振動データ
    
    client = mqtt.Client()
    client.connect(BROKER, 1883, 60)
    
    print("Start sending data...")
    num_chunks = len(vibration_data) // WINDOW_SIZE
    
    for i in range(num_chunks):
        start_idx = i * WINDOW_SIZE
        end_idx = start_idx + WINDOW_SIZE
        chunk = vibration_data[start_idx:end_idx].tolist()
        
        # JSON形式でパブリッシュ
        payload = json.dumps({"data": chunk})
        client.publish(TOPIC, payload)
        
        print(f"Sent chunk {i+1}/{num_chunks}")
        time.sleep(0.1)  # 0.1秒待機（ここがリアルタイムの鍵）

    client.disconnect()

if __name__ == "__main__":
    start_sender()