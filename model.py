from mlxtend.regressor import StackingRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_curve,auc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import torch
from model2 import IdentifyModel
import LIME
import LIME.lime_tabular

# 加载数据集
data = pd.read_csv('./FirstCleaned_Dataset.csv')

# 假设你的数据集有两列：'features'和'target'
X = data[['就诊次数_SUM','月统筹金额_MAX','ALL_SUM','月药品金额_AVG','可用账户报销金额_SUM']]
y = data['RES']  # 目标列，即你想要预测的列
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# 初始化SMOTE
smote = SMOTE(sampling_strategy='not majority',random_state=42)
# 拟合并转换数据
X_train, y_train = smote.fit_resample(X_train, y_train)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练基础模型并获取预测结果
train_prediction = np.zeros((X_train.shape[0], 3))
test_prediction = np.zeros((X_test.shape[0], 3))
# 创建LightGBM数据集
data0 = pd.read_excel('./result0.xlsx')
X0 = data[['就诊次数_SUM','月统筹金额_MAX','ALL_SUM','月药品金额_AVG','可用账户报销金额_SUM']]
y0 = pd.Series([0 for _ in range(20000)])
trainset = lgb.Dataset(X_train, label=y_train)
validationset = lgb.Dataset(X_test, label=y_test)
# 设置LightGBM参数（二分类）
params = {
    'objective': 'binary',
    'learning_rate': 0.02
}
# 训练模型
clf = lgb.train(params, trainset, valid_sets=[validationset],num_boost_round=100)
y_pred_prob = clf.predict(X_test)
y_pred = [1 if x > 0.5 else 0 for x in y_pred_prob]
test_prediction[:, 0] = y_pred

xgb = XGBClassifier(n_estimators = 50,learning_rate = 0.05,max_depth = 5,random_state=42)
# model = XGBClassifier(n_estimators = 125,learning_rate = 0.2,max_depth = 9)
xgb.fit(X_train,y_train)
y_pred_prob = xgb.predict(X_test)
test_prediction[:, 1] = y_pred_prob

MODEL_LOAD = "./model_final_0.79.pth"
mlp = IdentifyModel()
mlp.load_state_dict(torch.load(MODEL_LOAD))
with torch.no_grad():
    mlp.eval()
    x_test = X_test
    test_prediction[:, 2] = torch.argmax(mlp(torch.tensor(x_test.values, dtype=torch.float32)), dim=1)

# 初始化KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_prediction2 = np.zeros((5, X_test.shape[0], 3))
for i,(train_index, test_index) in enumerate(kf.split(X_train)):
    print(test_index)
    xtrain, xtest = X_train.iloc[train_index], X_train.iloc[test_index]
    ytrain, ytest = y_train.iloc[train_index], y_train.iloc[test_index]
    # 创建LightGBM数据集
    trainset = lgb.Dataset(xtrain, label=ytrain)
    validationset = lgb.Dataset(xtest, label=ytest)
    # 训练模型
    y_train_pred_prob = clf.predict(xtest)
    y_train_pred = [1 if x > 0.5 else 0 for x in y_train_pred_prob]
    train_prediction[test_index, 0] = y_train_pred

    y_train_pred = xgb.predict(xtest)
    train_prediction[test_index, 1]= y_train_pred

    y_train_pred = torch.argmax(mlp(torch.tensor(xtest.values, dtype=torch.float32)), dim=1)
    train_prediction[test_index, 2] = y_train_pred
    #测试集
    y_pred_prob = clf.predict(X_test)
    y_pred = [1 if x > 0.5 else 0 for x in y_pred_prob]
    test_prediction2[i,:,0] = y_pred

    y_pred_prob = xgb.predict(X_test)
    test_prediction2[i, :, 1] = y_pred_prob

    y_pred_prob = torch.argmax(mlp(torch.tensor(X_test.values, dtype=torch.float32)), dim=1)
    test_prediction2[i, :, 2] = y_pred_prob

test_prediction = test_prediction2.mean(axis=0)
print(train_prediction)
print(test_prediction)
# 将基础模型的预测结果堆叠起来，形成新的训练集和测试集
X_train_meta = train_prediction
X_test_meta = test_prediction
# 定义元模型
# 创建LightGBM数据集
trainset = lgb.Dataset(X_train_meta, label=y_train)
validationset = lgb.Dataset(X_test_meta, label=y_test)

# 设置LightGBM参数（二分类）
params = {
    'objective': 'binary',
    'learning_rate': 0.001
}

# 训练元模型
meta_model = lgb.train(params, trainset, valid_sets=[validationset],num_boost_round=5)

# 使用元模型进行预测
y_pred_prob = meta_model.predict(X_test_meta)
print(y_pred_prob)
# 将预测的概率转换为类别（大于0.5为正类，否则为负类）
y_pred = [1 if x > 0.5 else 0 for x in y_pred_prob]

# 计算并打印均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

prediction = y_pred
# 输出预测结果和准确率
print(y_pred)
y_pred = [1 if x > 0.5 else 0 for x in y_pred]
print('准确率:'+str(accuracy_score(y_test, y_pred)))


def calculate_negative_accuracy(y_true, y_pred):
    # 初始化计数器
    true_negatives = 0
    false_positives = 0
    total_negatives = sum(y_true == 1)  # 计算实际负样本总数
    # 遍历每个样本的真实标签和预测标签
    y_true = y_true.tolist()
    for i in range(len(y_true)):
        if y_true[i] == 1:  # 只关注负样本
            if y_pred[i] == 1:  # 预测也为负样本
                true_negatives += 1
            else:  # 预测为正样本，是错误分类
                false_positives += 1

                # 检查是否有实际负样本
    if total_negatives == 0:
        return 0  # 或者可以抛出一个错误或返回一个特殊值

    # 计算负样本的正确率
    negative_accuracy = true_negatives / total_negatives
    return negative_accuracy

# 计算并打印负样本的正确率
negative_accuracy = calculate_negative_accuracy(y_test, y_pred)
print("召回率:", negative_accuracy)

# 计算ROC曲线的真正类率和假正类率
fpr, tpr, thresholds = roc_curve(y_test, prediction)

# 计算AUC值
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

import pickle


# 保存模型
with open('clf.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('xgb.pkl', 'wb') as f:
    pickle.dump(xgb, f)

# 保存模型
torch.save(clf,'D:/mod/clf.pth')
torch.save(xgb,'D:/mod/xgb.pth')
torch.save(mlp,'D:/mod/mlp.pth')
torch.save(meta_model,'D:/mod/meta_model.pth')