from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE

def load_data(file_path: str):
    '''读取Excel'''
    df = pd.read_excel(file_path)
    df = df.dropna()  # 去除NaN值
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    return X, y

def save_data(file_path: str,X:pd.DataFrame, y:pd.Series):
    df = pd.concat([X, y], axis=1)
    df.to_excel(file_path, index=True)
    
def splitData(file_path: str,output_train_path='trainData.xlsx',output_test_path='testData.xlsx'):
    '''读取原始数据集并划分为训练集和测试集'''
    top_5_features = ['就诊次数_SUM', '月统筹金额_MAX', 'ALL_SUM','月药品金额_AVG', '可用账户报销金额_SUM']
    # 选择这5个特征
    
    df = pd.read_csv(file_path)
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    X = X[top_5_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    save_data(output_train_path,X_train,y_train)
    save_data(output_test_path,X_test,y_test)
    
def SMOTEOverSampling(X:pd.DataFrame, y:pd.Series):
    """对1进行SMOTE过采样"""
    # 创建 SMOTE 对象
    kmeans_smote = SMOTE(sampling_strategy='not majority', random_state=42)

    # 对数据进行过采样
    X_resampled, y_resampled = kmeans_smote.fit_resample(X, y)

    # 确认过采样后的样本数量
    print(f"过采样后的样本数：{len(X_resampled)}")

    # 确认过采样后 0 和 1 的比例
    print(f"过采样后标签为 0 的比例：{sum(y_resampled == 0) / len(y_resampled):.2f}")
    print(f"过采样后标签为 1 的比例：{sum(y_resampled == 1) / len(y_resampled):.2f}")
    return X_resampled, y_resampled

def step1():
    splitData('FirstCleaned_Dataset.csv')

def step2():
    X_train_org,y_train_org = load_data('trainData.xlsx')
    X_train_over,y_train_over = SMOTEOverSampling(X_train_org,y_train_org)
    save_data('trainData_SMOTE.xlsx',X_train_over,y_train_over)

step1()
step2()