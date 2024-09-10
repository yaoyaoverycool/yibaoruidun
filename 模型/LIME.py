import numpy as np
import lime
import lime.lime_tabular
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv('./FirstCleaned_Dataset.csv')

# 提取特征和目标列
X = data[['就诊次数_SUM','月统筹金额_MAX','ALL_SUM','月药品金额_AVG','可用账户报销金额_SUM']]
y = data['RES']

# 初始化SMOTE
smote = SMOTE(sampling_strategy='not majority', random_state=42)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用SMOTE进行过采样
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 创建LightGBM数据集
trainset = lgb.Dataset(X_train_resampled, label=y_train_resampled)
validationset = lgb.Dataset(X_test, label=y_test)

# 设置LightGBM参数
params = {
    'objective': 'binary',
    'learning_rate': 0.02
}

# 训练模型
clf = lgb.train(params, trainset, valid_sets=[validationset], num_boost_round=100)

# 创建特征名称列表
feature_names = X.columns.tolist()

# 定义可解释模型
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_resampled.values,
                                                   feature_names=feature_names,
                                                   class_names=['0', '1'],
                                                   discretize_continuous=True)

# 选择一个要解释的示例
print(X_test)
instance_idx = 16 #16
instance = X_test.iloc[instance_idx]
print(y_test.iloc[instance_idx])
# 定义一个函数来预测概率
def predict_proba(X):
    y_pred_prob_raw = clf.predict(X, raw_score=True)
    y_pred_prob = 1 / (1 + np.exp(-y_pred_prob_raw))  # 使用 sigmoid 函数将原始得分转换为概率
    return np.column_stack((1 - y_pred_prob, y_pred_prob))

# 将示例转换为一个数组
instance_array = np.array([instance.values])
print(instance_array)
# 使用LIME解释模型
explanation = explainer.explain_instance(instance.values,
                                         predict_proba,
                                         num_features=len(feature_names))

# 打印解释结果
print("预测:", clf.predict(instance_array)[0])

print("解释:")
for i in range(len(feature_names)):
    print(f"{feature_names[i]}: {explanation.local_exp[1][i][1]}  {explanation.local_exp[1][i][0]}")
# 导入必要的库
import matplotlib.pyplot as plt
# 打印预测分数
prediction_score = clf.predict(instance_array)[0]
print("预测分数:", prediction_score)

# 获取特征重要性
sorted_features = explanation.as_list()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 解释结果可视化
feature_names, feature_values = zip(*sorted_features)
print(feature_values,type(feature_values))
# Define colors for positive and negative feature importances
colors = ['darkorange' if value > 0 else 'blue' for value in feature_values]
feature_values = tuple(-x for x in feature_values)
plt.barh(range(len(feature_names)), feature_values,color=colors, align='center')
# 在每个柱形上方添加文本显示特征重要性值
for i, value in enumerate(feature_values):
    if value>0:
        plt.text(value, i, f' {value:.2f}', ha='left', va='center', color='black')
    else:
        plt.text(value, i, f' {value:.2f}', ha='right', va='center', color='black')
plt.axvline(x=prediction_score, color='gray', linestyle='', linewidth=1)  # 垂直线表示预测分数
# 添加正负类预测分数标签
plt.text(prediction_score + 0.05, len(feature_names) - 0.6, f'正类预测分数: {1 - prediction_score:.2f}', color='blue',fontsize=14, fontweight='bold')
plt.text(prediction_score + 0.05, len(feature_names) - 0.9, f'负类预测分数: {prediction_score:.2f}', color='darkorange',fontsize=14, fontweight='bold')
#plt.text(prediction_score - 0.3, len(feature_names) - 0.6, f'正类预测分数: {1 - prediction_score:.2f}', color='blue',fontsize=14, fontweight='bold')
#plt.text(prediction_score - 0.3, len(feature_names) - 0.9, f'负类预测分数: {prediction_score:.2f}', color='darkorange',fontsize=14, fontweight='bold')

plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('特征重要性')
plt.title(f'LIME对测试集第{instance_idx}个样本预测结果的决策解释')
plt.tight_layout()  # 调整子图参数，确保内容完整显示
plt.show()
