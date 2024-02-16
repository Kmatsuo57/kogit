# improt library

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from stardist.models import StarDist2D
from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd




def calculate_fluorescence(roi):
    mean_value = np.mean(roi)
    std_dev = np.std(roi)
    return mean_value, std_dev



# 四分位数を使用して外れ値を除外する関数
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def process_directory(input_directory):

    image_files = sorted([f for f in os.listdir(input_directory) if f.endswith('.tif')])
    # Loading Pre-trained model

    std_devs = []
    mean_values = []
    green_cells = []
    merged_cells = []
    all_data = []
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    all_raw_counts = 0
    for i, image_file in enumerate(image_files):

        normalized_areas = []  # 正規化された面積を保存するリスト
        circularities = []
        fluorescence_values = []
        # # test image loading
        # chlamy_image_path = '/Users/Applied_Mol_Microbiol/Documents/image_data/240206_RV5ccm1_LC_pyrearea/RV5ccm1_LC.lif_02.tif'
        image_path = os.path.join(input_directory, image_file)
        chlamy_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if chlamy_image is None:
            print(f'画像の読み込みができませんでした{i}')
        else:
            print(f'読み込みに成功しました。解析を開始します。{i}')
        red_image = chlamy_image[:,:, 2]

        # show test image and prediction

        labels, details = model.predict_instances(normalize(red_image))

        plt.figure()

        plt.subplot(1,2,1)
        plt.imshow(red_image, cmap='gray')
        plt.axis('off')
        plt.title('input image')

        plt.subplot(1,2,2)
        plt.imshow(render_label(labels, img=red_image))
        plt.axis('off')
        plt.title('prediction + input image overlay')

        plt.show(block=False)


        # 各細胞に対する処理

        

        for label in np.unique(labels):
            if label == 0:
                continue  # 背景をスキップ

            # 細胞のマスクを作成
            cell_mask = labels == label

            # グリーンチャネルの抽出
            green_image = chlamy_image[:,:,1]
            
            # 細胞のマスクを適用
            green_cell = np.where(cell_mask, green_image, 0)
            # 各細胞に対する処理のループ内...
            green_cells.append(green_cell)
            # 各細胞に対する処理のループ内...
            # cell_mask を3次元に拡張
            cell_mask_3d = np.repeat(cell_mask[:, :, np.newaxis], 3, axis=2)

            # 細胞のマスクを適用
            merged_cell = np.where(cell_mask_3d, chlamy_image, 0)
            merged_cells.append(merged_cell)




            # グリーンチャネルの二値化（Otsuの方法）
            ret, thresh = cv2.threshold(green_cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # findContoursで輪郭を見つける
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 最大の輪郭を見つける
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                # 最大の輪郭の面積を計算
                largest_area = cv2.contourArea(largest_contour)
                if largest_area != 0:
                    
                    perimeter = cv2.arcLength(largest_contour, True)


                    # 細胞の面積で正規化
                    cell_area = np.sum(cell_mask)
                    normalized_area = largest_area / cell_area

                    circularity = 4 * np.pi * largest_area / (perimeter ** 2)

                    mean_value = np.mean(green_cell)
                    std_dev = np.std(green_cell)
                    # 結果をリストに保存
                    mean_values.append(mean_value)
                    std_devs.append(std_dev)
                    fluorescence_values.append((mean_value, std_dev))
                    
                    normalized_areas.append(normalized_area)
                    circularities.append(circularity)
            
                

        cvs = [std_dev / mean for mean, std_dev in fluorescence_values]

        # データをDataFrameに格納
        data = pd.DataFrame({
        'Normalized Area': normalized_areas,
        'Circularity': circularities,
        'CV': [std_dev / mean for mean, std_dev in fluorescence_values]
        })

        # CVとCircularityの外れ値を除外
        cleaned_data = remove_outliers(data, 'CV')
        cleaned_data = remove_outliers(cleaned_data, 'Circularity')
        cleaned_data = remove_outliers(cleaned_data, 'Normalized Area')

        raw_count = len(cleaned_data)
        all_raw_counts += raw_count
        print(f'{i}回目に解析した細胞数：{raw_count}個')
        # 画像ごとのクリーンなデータをリストに追加
        all_data.append(cleaned_data)

    # 全ての画像ファイルのデータを統合
    final_data = pd.concat(all_data)
    return final_data, all_raw_counts



input_directories = ['/Users/Applied_Mol_Microbiol/Documents/image_data/240206_RV5_LC_pyrearea/', '/Users/Applied_Mol_Microbiol/Documents/image_data/240206_RV5ccm1_LC_pyrearea/']


data_red, raw_counts_red = process_directory(input_directories[0])
data_blue, raw_counts_blue = process_directory(input_directories[1])

print(f'{raw_counts_red}個の細胞のデータを解析しました。')
print(f'{raw_counts_blue}個の細胞のデータを解析しました。')

# 3D散布図にデータをプロット
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

# 1つ目のディレクトリ（赤色）
ax.scatter(data_red['Normalized Area'], data_red['Circularity'], data_red['CV'], c='r', marker='o', label='RV')

# 2つ目のディレクトリ（青色）
ax.scatter(data_blue['Normalized Area'], data_blue['Circularity'], data_blue['CV'], c='b', marker='o', label='ccm1 mutant')

ax.set_xlabel('Normalized Area')
ax.set_ylabel('Circularity')
ax.set_zlabel('CV')
ax.legend()

plt.show()

# def update(frame):
#     ax.view_init(elev=10., azim=frame)
#     return fig,

# # アニメーションの作成
# ani = FuncAnimation(fig, update, frames=range(0, 360, 1), blit=True)

# # アニメーションを保存（mp4ファイルとして）
# ani.save('rotation_animation.mp4', writer='ffmpeg', fps=30)






# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')

# ax.scatter(final_data['Normalized Area'], final_data['Circularity'], final_data['CV'], c='r', marker='o')
# ax.set_xlabel('Normalized Area')
# ax.set_xlabel('Normalized Area')
# ax.set_ylabel('Circularity')
# ax.set_zlabel('CV')

# plt.show()

# print(fluorescence_values)
# print(len(normalized_areas))
# plt.scatter(normalized_areas,circularities, s = 10, c = 'k')
# plt.xlabel('related pyrenoid area')
# plt.ylabel('cirsulatiy')
# plt.show()


# # ビデオライターの初期化
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# # frameSize を取得
# height, width = green_cells[0].shape[:2]

# # fps (フレームレート) を設定
# fps = 5.0

# out = cv2.VideoWriter('green_cells_video.mp4', fourcc, fps, (width, height))

# # 各画像をビデオファイルに追加
# for img in green_cells:
#     # グレースケール画像をカラー画像に変換する（必要に応じて）
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     out.write(img)

# # ビデオライターのリリース
# out.release()

# # ビデオライターの初期化
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# # frameSize を取得（merged_cellsの最初の画像から）
# height, width = merged_cells[0].shape[:2]

# # fps (フレームレート) を設定
# fps = 5.0

# out_merged = cv2.VideoWriter('merged_cells_video.mp4', fourcc, fps, (width, height))

# # 各 `merged_cell` 画像をビデオファイルに追加
# for img in merged_cells:
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     out_merged.write(img)

# # ビデオライターのリリース
# out_merged.release()

# current_direcrory = os.getcwd()
# print(f'動画を{current_direcrory}へ出力しました')
