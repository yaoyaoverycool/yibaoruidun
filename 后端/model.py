import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split

clf = torch.load('./clf.pth')
xgb = torch.load('./xgb.pth')
mlp = torch.load('./mlp.pth')
meta_model = torch.load('./meta_model.pth')


# 加载数据集
data = pd.read_csv('./FirstCleaned_Dataset.csv')

# 假设你的数据集有两列：'features'和'target'
# X = data[['就诊次数_SUM', '月统筹金额_MAX', 'ALL_SUM','民政救助补助_SUM', '城乡救助补助金额_SUM']]
X = data[['就诊次数_SUM','月统筹金额_MAX','ALL_SUM','月药品金额_AVG','可用账户报销金额_SUM']]
y = data['RES']  # 目标列，即你想要预测的列
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,random_state=45)

test_prediction = np.zeros((X_test.shape[0], 3))

y_pred_prob = clf.predict(X_test)
y_pred = [1 if x > 0.5 else 0 for x in y_pred_prob]
test_prediction[:, 0] = y_pred

y_pred_prob = xgb.predict(X_test)
test_prediction[:, 1] = y_pred_prob

y_pred_prob = torch.argmax(mlp(torch.tensor(X_test.values, dtype=torch.float32)), dim=1)
test_prediction[:, 2] = y_pred_prob

# 使用元模型进行预测
y_pred = meta_model.predict(test_prediction)
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
    true0 = 0
    true1 = 0
    total0 = sum(y_true == 0)  # 类别0的负样本总数
    total1 = sum(y_true == 1)  # 类别1的负样本总数
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    true0 = np.sum((y_true_np == 0) & (y_pred_np == 0))
    true1 = np.sum((y_true_np == 1) & (y_pred_np == 1))

    # 计算准确率
    print(total0)
    print(total1)
    accuracy0 = true0 / total0 if total0 != 0 else 0
    accuracy1 = true1 / total1 if total1 != 0 else 0
#    print(total_negatives)
#    print(len(y_true))
#    print(len(y_true.tolist()))
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
    return negative_accuracy, accuracy0, accuracy1

# 计算并打印负样本的正确率
negative_accuracy, acc0, acc1 = calculate_negative_accuracy(y_test, y_pred)
print("召回率:", negative_accuracy)
print("类别0准确率:", acc0)
print("类别1准确率:", acc1)

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