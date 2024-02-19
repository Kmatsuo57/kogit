import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

directory = '/Users/Applied_Mol_Microbiol/Desktop/python_data/'
file_name = 'RV5_LC_CV.xlsx'
file_path = os.path.join(directory, file_name)

# カラム名のリストを生成
columns_to_read = [f'RV5_LC{i}' for i in range(1, 7)]

# Excelファイルを読み込み、特定のカラムのみを選択
data = pd.read_excel(file_path, usecols=columns_to_read)

def standardize(series):
    return (series - series.mean()) / series.std()

# 各カラムを標準化
standardized_data = data.apply(standardize)

trimmed_data_list = []

for column in standardized_data.columns:
    trim_data = standardized_data[column].dropna(axis=0, how='any')
    trimmed_data_list.append(trim_data)

def quadratic_solver(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant >= 0:
        x1 = (-b + np.sqrt(discriminant)) / (2*a)
        x2 = (-b - np.sqrt(discriminant)) / (2*a)
        return x1, x2
    else:
        return None  # 実数解が存在しない場合

# 二次関数にフィットさせたデータを格納するリスト
quadratic_data_list = []
new_df = pd.DataFrame()

for j in range(len(trimmed_data_list)):
    series = trimmed_data_list[j]
    x = np.arange(len(series))
    # print(len(series))
    # 二次関数にフィット
    coefficients = np.polyfit(x, series, 2)
    quadratic_values = np.polyval(coefficients, x)
    
    # フィットしたデータをリストに追加
    quadratic_data_list.append(pd.Series(quadratic_values, index=series.index))
    
    # 二次関数の式を表示
    polynomial_equation = np.poly1d(coefficients)
    a, b, c = coefficients
    
    min_sdf = standardized_data[f'RV5_LC{j+1}'].min()
    max_sdf = standardized_data[f'RV5_LC{j+1}'].max()
    middle_value = (max_sdf - min_sdf) / 2 + min_sdf
    print(middle_value)
    # 二次方程式の解を求める
    solutions = quadratic_solver(a, b, c - middle_value - 0.2)

    if solutions:
        x1, x2 = solutions
        if 0 < x1 < len(series) and 0 < x2 < len(series):
            # print(f"Debug - j={j}, x1={x1}, x2={x2}, len(series)={len(series)}")
            new_df[f'RV5_LC{j+1}'] = standardized_data[f'RV5_LC{j+1}'].iloc[int(x2):int(x1)].reset_index(drop=True)
        else:
            pass
    else:
        print(f"No real solutions for Column {j+1}")
column_to_exclude = ['RV5_LC2']
trimmed_data = new_df.drop(columns=column_to_exclude)


plt.figure(figsize=(10, 2))
sns.heatmap(new_df.T, annot=False, cmap='coolwarm')
# plt.plot(new_df)
# for column in trimmed_data.columns:
#     plt.plot(trimmed_data[column], label=column)
# plt.ylim(-2.5, 2.5)
plt.title('Heatmap of Data Around Minimum Values in Each Column')
plt.yticks([])
plt.xticks([])
plt.xlabel('Time (rows)')
plt.ylabel('Samples (columns)')
# plt.legend()
plt.show()

# sns.heatmap(trimmed_data.T, annot=False, cmap='coolwarm')
# plt.show()

# # プロット
# plt.figure(figsize=(10, 6))
# for j in range(len(quadratic_data_list)):
#     plt.plot(quadratic_data_list[j], label=f'Column {j + 1}')

# plt.xlabel('Index')
# plt.ylabel('Quadratic Values')
# plt.legend()
# plt.show()
