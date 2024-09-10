from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

# 假设 self.X_train, self.X_test, self.y_train, self.y_test 已经正确加载和划分

# 加载数据集
data = pd.read_csv('./FirstCleaned_Dataset.csv')

# 假设你的数据集有两列：'features'和'target'
X = data[['月就诊次数_MAX', '医院_统筹金_MAX', '月药品金额_MAX','ALL_SUM', '顺序号_NN']]
y = data['RES']  # 目标列，即你想要预测的列

# 初始化SMOTE
smote = SMOTE(sampling_strategy='not majority',random_state=42)
#adasyn = ADASYN(sampling_strategy='not majority', random_state=42)


# 划分数据集为训练集和测试集，通常测试集占20%-30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# 拟合并转换数据
X_train, y_train = smote.fit_resample(X_train, y_train)
#X_train, y_train = adasyn.fit_resample(X_train, y_train)

# 初始化欠采样器
under_sampler = RandomUnderSampler(random_state=42)

# 进行欠采样
X_train, y_train = under_sampler.fit_resample(X_train, y_train)


# 特征缩放

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建LightGBM数据集
trainset = lgb.Dataset(X_train, label=y_train)
validationset = lgb.Dataset(X_test, label=y_test)

# 设置LightGBM参数（二分类）
params = {
    'objective': 'binary',
    'learning_rate': 0.02
}

# 训练模型
clf = lgb.train(params, trainset, valid_sets=[validationset],num_boost_round=100)

# 预测
y_pred_prob = clf.predict(X_test)
print(y_pred_prob)
# 将预测的概率转换为类别（大于0.5为正类，否则为负类）
y_pred = [1 if x > 0.5 else 0 for x in y_pred_prob]

# 输出预测结果和准确率
print(y_pred)
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