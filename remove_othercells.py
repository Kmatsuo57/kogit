import cv2
import numpy as np
import os
import glob
def remove_cells_at_edges(image, buffer_size):#buffer_sizeは画像の4辺からの距離.10を入力したら4辺から10pix内にある細胞を削除してくれる
  

    # Redチャンネルのみを取得
    red_channel = image[:,:,2]

    # 二値化
    _, binary_image = cv2.threshold(red_channel, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 輪郭を見つける
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 画像のサイズ
    height, width = image.shape[:2]

    for contour in contours:
        # 輪郭のバウンディングボックスを取得
        x, y, w, h = cv2.boundingRect(contour)

        # バッファゾーン内にあるかどうかチェック
        if x <= buffer_size or y <= buffer_size or (x+w) >= (width - buffer_size) or (y+h) >= (height - buffer_size):
            # バッファゾーン内の輪郭を黒色で塗りつぶす
            cv2.drawContours(image, [contour], -1, (0, 0, 0), -1)

    return image



def process_tiff_sequence(input_folder, output_folder, buffer_size):
    # 出力フォルダがなければ作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 入力フォルダからTIFFファイルのリストを取得
    image_files = glob.glob(os.path.join(input_folder, "*.tif"))
    sorted_image_files = sorted(image_files)


    for idx, file in enumerate(sorted_image_files):
        # 画像を読み込む
        image = cv2.imread(file)
        if image is None:
            continue

        # 画像を処理
        processed_image = remove_cells_at_edges(image, buffer_size)

        # 処理された画像を保存
        output_path = os.path.join(output_folder, f"processed_frame_{idx}.tif")
        cv2.imwrite(output_path, processed_image)



input_dir = "/Users/Applied_Mol_Microbiol/Documents/image_data/RV5ccm1/LC/Non_treat/"
output_dir = "/Users/Applied_Mol_Microbiol/Documents/image_data/RV5ccm1/LC/Processed_image/"

for i in range(1, 5):
    folder_name = f"RV5ccm1_LC{i}"
    input_folder = input_dir + folder_name
    output_folder = output_dir + folder_name
    process_tiff_sequence(input_folder, output_folder, 10)



