@echo off
setlocal

:: 1. Mosquitto を起動
wt -w 0 nt --title "Mosquitto" cmd /k " \"C:\Program Files\Mosquitto\mosquitto.exe\" -v "

timeout /t 3 /nobreak > nul

:: 2. Receiver (仮想環境を有効化してから実行)
:: ポイント：cmd /k "コマンド" の形にし、中のパスは \" で囲みます
wt -w 0 split-pane -V --title "Receiver" cmd /k " call ..\venv\Scripts\activate && python .\main_receiver.py"

timeout /t 3 /nobreak > nul

:: 3. Sender
wt -w 0 split-pane -H --title "Sender" cmd /k " call ..\venv\Scripts\activate && python .\sender_sim.py"

endlocal