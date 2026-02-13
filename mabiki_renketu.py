import numpy as np
import os
import csv

def process_and_save_waveform(input_file, num_repeats=200):
    print(f"Reading {input_file}...")
    
    raw_data = []
    is_data_section = False
    
    # --- 1. データ読み込み (1回分) ---
    with open(input_file, mode='r', encoding='shift_jis') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            tag = row[0].strip()
            if tag == "#EndHeader":
                is_data_section = True
                continue
            elif tag == "#BeginMark":
                break
            if is_data_section:
                try:
                    raw_data.append(float(row[2]))
                except (ValueError, IndexError):
                    continue

    if not raw_data:
        print("Error: Could not find valid data.")
        return

    # --- 2. 25kHzへのダウンサンプリング (50kHz -> 25kHz) ---
    raw_signal = np.array(raw_data)
    if len(raw_signal) % 2 != 0:
        raw_signal = raw_signal[:-1]
    resampled_single = (raw_signal[0::2] + raw_signal[1::2]) / 2
    
    single_len = len(resampled_single)
    dt = 1.0 / 25000  # 0.00004秒

    # --- 3. 追記モードで保存 ---
    output_filename = "../new_waveform_25khz.csv"
    output_path = os.path.join(os.path.dirname(os.path.abspath(input_file)), output_filename)
    
    # 既存ファイルを削除してから開始
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Starting incremental save to {output_filename}...")
    
    current_total_samples = 0
    
    # 1200回繰り返しながら、その都度ファイルに書き込む
    # ※クロスフェードは今回は簡易化のため省略し、単純連結します。
    # (高周波サンプリングでは単純連結でもスコアへの影響は限定的です)
    with open(output_path, mode='a', newline='') as f_out:
        writer = csv.writer(f_out)
        
        for i in range(num_repeats):
            if i % 100 == 0:
                print(f"Processing... {i}/{num_repeats}")
            
            # このブロック用の時間軸を作成
            times = (np.arange(single_len) + current_total_samples) * dt
            
            # [時間, 振動値] のペアをリストにして一括書き込み
            chunk = np.column_stack((times, resampled_single))
            writer.writerows(chunk)
            
            current_total_samples += single_len

    print(f"Successfully completed! Total samples: {current_total_samples}")

# 実行
input_csv = '../20260113passなし1000rpm300Nm.CSV'
process_and_save_waveform(input_csv)