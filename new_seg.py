import cv2
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

def load_image_and_annotation(image_path, json_path):
    # グレースケール画像として読み込み
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"画像を読み込めませんでした: {image_path}")
        return None, None

    # 必要に応じてRGBに変換
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (128, 128))

    # JSONファイルからアノテーションの読み込み
    with open(json_path, 'r') as file:
        data = json.load(file)
    polygons = [shape['points'] for shape in data['shapes']]

    labels = [shape['label'] for shape in data['shapes']]
    return image, polygons, labels


def flip_image_and_annotation(image, polygons):
    flipped_image = cv2.flip(image, 1)
    flipped_polygons = []
    for polygon in polygons:
        flipped_polygon = [[image.shape[1] - p[0], p[1]] for p in polygon]
        flipped_polygons.append(flipped_polygon)

    return flipped_image, flipped_polygons


def rotate_image_and_annotation(image, polygons, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    rotated_polygons = []
    for polygon in polygons:
        rotated_polygon = []
        for p in polygon:
            # ポイントを2Dから3Dに拡張
            p_3d = np.array([p[0], p[1], 1])
            # 回転行列を適用
            rotated_p = np.dot(M, p_3d)
            rotated_polygon.append([rotated_p[0], rotated_p[1]])
        rotated_polygons.append(rotated_polygon)

    return rotated_image, rotated_polygons



def scale_image_and_annotation(image, polygons, scale_factor):
    (h, w) = image.shape[:2]
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    scaled_polygons = []
    for polygon in polygons:
        scaled_polygon = [[p[0] * scale_factor, p[1] * scale_factor] for p in polygon]
        scaled_polygons.append(scaled_polygon)

    return scaled_image, scaled_polygons


def vertical_flip_image_and_annotation(image, polygons):
    flipped_image = cv2.flip(image, 0)
    flipped_polygons = []
    for polygon in polygons:
        flipped_polygon = [[p[0], image.shape[0] - p[1]] for p in polygon]
        flipped_polygons.append(flipped_polygon)

    return flipped_image, flipped_polygons


def convert_polygons_to_mask(polygons, labels, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for polygon, label in zip(polygons, labels):
        if len(polygon) < 3:
            continue
        int_polygon = np.array(polygon, dtype=np.int32)
        print(int_polygon.shape)
        mask_value = 1 if label == 'Starch plate' else 2 if label == 'Starch glanule' else 0
        cv2.fillPoly(mask, [int_polygon], color=mask_value)
    return mask


def extend_images(image_path, json_path, degree):
    image, polygon, labels = load_image_and_annotation(image_path, json_path)
    # フリップされた画像とアノテーションを取得
    flipped_image, flipped_polygons = flip_image_and_annotation(image, polygon)
    print(len(flipped_image))
    rotate_image, rotate_polygons = rotate_image_and_annotation(image, polygon, degree)
    print(len(rotate_image))
    vertical_flip_image, vertical_flip_polygons = vertical_flip_image_and_annotation(image, polygon)
    print(len(vertical_flip_image))


    images = []
    polygons = []

    # 元の画像と水増し画像をリストに追加
    images.append(image)
    polygons.append(polygon)

    # 水増し画像をリストに追加
    images.extend([flipped_image, rotate_image, vertical_flip_image])
    polygons.extend([flipped_polygons, rotate_polygons, vertical_flip_polygons])

    # NumPy配列に変換
    images = np.array(images)

    # ここでポリゴンをセグメンテーションマスクに変換する関数を定義する必要があります
    masks = [convert_polygons_to_mask(p, l, image.shape) for p, l in zip(polygons, labels)]
    masks = np.array(masks)

    return images, masks

def load_image(image_path):
    # グレースケール画像として読み込み
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"画像を読み込めませんでした: {image_path}")
        return None

    # 必要に応じてRGBに変換
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (128, 128))

    return image

def segment_image(model, image_path):
    # 画像の読み込みと前処理
    image = load_image(image_path)
    if image is None:
        print("画像が読み込めませんでした。")
        return None

    image = np.expand_dims(image, axis=0)  # モデルの入力形状に合わせる

    # 予測
    predicted_mask = model.predict(image)

    return predicted_mask[0]

class VisualizeCallback(Callback):
    def __init__(self, image, mask):
        super(VisualizeCallback, self).__init__()
        self.image = image
        self.mask = mask

    def on_epoch_end(self, epoch, logs=None):
        # 予測されたマスクを取得
        predicted_mask = self.model.predict(np.expand_dims(self.image, axis=0))[0].squeeze()

        # 元の画像、真のマスク、予測されたマスクを表示
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(self.image.squeeze(), cmap='gray')
        plt.title("Original Image")

        plt.subplot(1, 3, 2)
        plt.imshow(self.mask.squeeze(), cmap='gray')
        plt.title("True Mask")

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title("Predicted Mask")

        plt.show()



image_directory = '/Users/Applied_Mol_Microbiol/Documents/python_data/teacher_img/'
json_directory = '/Users/Applied_Mol_Microbiol/Documents/python_data/python_396/RV5_teacher/json_data/'

image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.png')])
json_files = sorted([f for f in os.listdir(json_directory) if f.endswith('.json')])



print('image_filesの長さ', len(image_files))
print('json_filesの長さ',len(json_files))

df4segment = pd.DataFrame({'images': image_files, 'masks': json_files})

print(df4segment)

images = []
masks = []

for _, row in df4segment.iterrows():
    extended_image, extended_mask = extend_images(os.path.join(image_directory, row['images']), os.path.join(json_directory, row['masks']), 90)
    images.append(extended_image)
    masks.append(extended_mask)

# print(images, masks)


from sklearn.model_selection import train_test_split

# トレーニングデータとテストデータに分割（例えば、80%トレーニング、20%テスト）
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

# さらにトレーニングデータをトレーニングセットと検証セットに分割（例えば、80%トレーニング、20%検証）
train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)

print(len(train_images), len(test_images), len(train_masks), len(test_masks))
resized_train_masks = [cv2.resize(mask, (128, 128)) for mask in train_masks]

# テストデータのマスクをリサイズ
resized_test_masks = [cv2.resize(mask, (128, 128)) for mask in test_masks]

train_images_stacked = np.stack(train_images)
train_masks_stacked = np.stack(train_masks)

test_images_stacked = np.stack(test_images)
test_masks_stacked = np.stack(test_masks)

train_images_concatenated = np.concatenate(train_images)
train_masks_concatenated = np.concatenate(train_masks)

test_images_concatenated = np.concatenate(test_images)
test_masks_concatenated = np.concatenate(test_masks)

# print('train_imageのサイズ:',train_images.shape)
# print('train_masksのサイズ:', train_masks.shape)

print('train_images_concatenatedのサイズ:',train_images_concatenated.shape)
print('train_masks_concatenatedのサイズ:', train_masks_concatenated.shape)
print('test_images_concatenatedのサイズ:', test_images_concatenated.shape)
print('test_masks_concatenatedのサイズ:', test_masks_concatenated.shape)




from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model

# エンコーダとしてMobileNetV2をロード
base_model = MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# 各レイヤーを凍結（重みを固定）
for layer in base_model.layers:
    layer.trainable = False

# デコーダの構築
inputs = Input(shape=[128, 128, 3])
x = base_model(inputs)
x = UpSampling2D()(x)  # 8x8x1280
x = Conv2D(256, 3, padding='same', activation='relu')(x)
x = UpSampling2D()(x)  # 16x16x256
x = Conv2D(128, 3, padding='same', activation='relu')(x)
x = UpSampling2D(size=(2, 2))(x)  # 32x32x128
x = UpSampling2D(size=(2, 2))(x)  # 64x64x128
x = UpSampling2D(size=(2, 2))(x)  # 128x128x128
outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(x)  # セグメンテーションマスクの出力



# モデルの定義
model = Model(inputs=inputs, outputs=outputs)

# モデルのコンパイル
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# バッチサイズとエポック数を設定
batch_size = 32
epochs = 10

# 画像とマスクを選択
sample_image = train_images_concatenated[0]
sample_mask = train_masks_concatenated[0]

print("マスクの値の範囲:", np.min(sample_mask), "～", np.max(sample_mask))
plt.imshow(sample_mask.squeeze(), cmap='gray', vmin=0, vmax=1)


# # コールバックを作成
# vis_callback = VisualizeCallback(sample_image, sample_mask)

# # モデルのトレーニング（コールバックを含む）
# model.fit(train_images_concatenated, train_masks_concatenated, 
#           epochs=10,
#           callbacks=[vis_callback],
#           validation_data=(val_images, val_masks))



# # テストセットでのモデルの評価
# test_loss, test_accuracy = model.evaluate(test_images_concatenated, test_masks_concatenated)
# print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# image_path = '/Users/Applied_Mol_Microbiol/Documents/python_data/python_396/RV5_LC_TEM/RV5_LC_009.bmp'

# predicted_mask = segment_image(model, image_path)

# if predicted_mask is not None:
#     plt.imshow(predicted_mask.squeeze(), cmap='gray')
#     plt.show()

