import cv2
import numpy as np

# 動画ファイルのパス
video_path = '/Users/Applied_Mol_Microbiol/Documents/240206_RV5ccm1_LC-1.avi'

# 動画を読み込む
cap = cv2.VideoCapture(video_path)

# 処理後の動画を保存するための設定
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 赤チャンネルを抽出
    red_channel = frame[:, :, 2]

    # 二値化
    _, binary = cv2.threshold(red_channel, 127, 255, cv2.THRESH_BINARY)

    # 輪郭を検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 各輪郭に対する矩形ROIを取得
        x, y, w, h = cv2.boundingRect(contour)

        # ROIが画像の端に接しているか確認
        if x <= 0 or y <= 0 or (x + w) >= frame_width or (y + h) >= frame_height:
            # ROI内の全てのチャンネルの輝度を0に設定
            frame[y:y+h, x:x+w] = 0

    # 処理後のフレームを動画に書き込む
    out.write(frame)

# 動画のリリース
cap.release()
out.release()
