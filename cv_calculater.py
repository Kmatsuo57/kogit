import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from datetime import date


def cv_calcuation(image_file):

    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED) 

    green_ch = image[:, :, 1]
    red_ch = image[:, :, 2]
    
    #画像の前処理　メジアンフィルターの適用, ガウシアンフィルタにより平滑化
    med_red_ch = cv2.medianBlur(red_ch, 5)
    gaussian_red_ch = cv2.GaussianBlur(med_red_ch, (5, 5), 0)

    #Otsuの２値化法
    _, binary_red_channel = cv2.threshold(gaussian_red_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #画像内の輪郭を見つける
    contours, _ = cv2.findContours(binary_red_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #ノイズ処理により除去できない輪郭を除去する。このとき観察対象より十分に小さい値を設定する。　
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= 500]

    contours_counts = len(valid_contours)
    all_cv = 0
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = green_ch[y:y+h, x:x+w]
        mean_value = np.mean(roi)
        std_dev = np.std(roi)
        if mean_value > 0 and std_dev >0:
            cv_ave = std_dev / mean_value
            all_cv += cv_ave
        
        else:
            pass
    
    cv = all_cv / contours_counts

    return cv


dir = "/Users/Applied_Mol_Microbiol/Documents/image_data/RV5ccm1/LC/Processed_image/"
strain_and_condition = 'RV5ccm1_LC*' #最後にアスタリスクを忘れないように

timelaps_directories = sorted([d for d in glob.glob(os.path.join(dir, strain_and_condition)) if os.path.isdir(d)])
all_cv_list = []
for timelaps_directory in timelaps_directories:

    image_files = sorted(glob.glob(os.path.join(timelaps_directory, '*.tif')))

    timecource_cv = []
    for image in image_files:
        contours_counts = cv_calcuation(image)
        timecource_cv.append(contours_counts)

    all_cv_list.append(timecource_cv)


cv_df = pd.DataFrame(all_cv_list).transpose()
output_path = dir + "/result"
if not os.path.exists(output_path):
        os.makedirs(output_path)

output_data = output_path + 'all_cv_by_py.csv'
cv_df.to_csv(output_data, index=False)


















    






