from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE



data = pd.read_csv('./FirstCleaned_Dataset.csv')

X = data[['就诊次数_SUM','月统筹金额_MAX','ALL_SUM','月药品金额_AVG','可用账户报销金额_SUM']]
y = data['RES']  # 目标列，即你想要预测的列

smote = SMOTE(sampling_strategy='not majority',random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = XGBClassifier(n_estimators = 50,learning_rate = 0.05,max_depth = 5,random_state=42)
# model = XGBClassifier(n_estimators = 125,learning_rate = 0.2,max_depth = 9)
model.fit(X_train,y_train)


y_pred = model.predict(X_test)
print(y_pred)
accuracy = accuracy_score(y_test,y_pred)
print('准确率:'+ str(accuracy))
recall = recall_score(y_test, y_pred)
print('召回率1:', recall)