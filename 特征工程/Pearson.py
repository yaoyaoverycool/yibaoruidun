import pandas as pd
import numpy as np

#加载数据集
data_path = r"C:\Users\huranran\Desktop\comp\Test_model\test.py\Dataset_origin.csv"
data = pd.read_csv(data_path)
#预处理
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# 计算特征与目标变量`RES`的相关性
correlation = data.corr()['RES'].sort_values(ascending=False)

# 选择与`RES`相关性最强的前30个特征（除去`RES`自身）
top_features = correlation.abs().sort_values(ascending=False).head(31).index.tolist()
top_features = [feature for feature in top_features if feature != 'RES']  # 排除目标变量本身

#print("Selected features:", top_features)

#保留目标变量
selected_features_data = data[top_features + ['RES']] 

output_file_path = r"C:\Users\huranran\Desktop\comp\Test_model\test.py\First_Selected_Features.csv"
selected_features_data.to_csv(output_file_path, index=False) 

