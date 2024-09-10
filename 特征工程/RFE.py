import pandas as pd
import numpy as np
#选择随机森林模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

data_path = r"C:\Users\huranran\Desktop\comp\Test_model\test.py\First_Selected_Features.csv"
data = pd.read_csv(data_path)

X = data.drop('RES', axis=1)  
y = data['RES']

base_model = RandomForestClassifier(random_state=42)

rfe = RFE(estimator=base_model, n_features_to_select=10)
rfe = rfe.fit(X, y)

selected_features = X.columns[rfe.support_]
#print("Selected features:", selected_features)

# 保留被RFE选择的特征以及目标变量'RES'
second_selected_features_data = data[selected_features.tolist() + ['RES']]
output_file_path = r"C:\Users\huranran\Desktop\comp\Test_model\test.py\Second_Selected_Features.csv"

second_selected_features_data.to_csv(output_file_path, index=False) 