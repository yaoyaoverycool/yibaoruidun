import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_path = r"C:\Users\huranran\Desktop\comp\Test_model\test.py\Second_Selected_Features.csv"
data = pd.read_csv(data_path)
data = pd.read_csv(data_path)

X = data.drop('RES', axis=1)  
y = data['RES']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_feature_set(features):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train[features], y_train)
    predictions = model.predict(X_test[features])
    return accuracy_score(y_test, predictions)

features = list(X.columns)
selected_features = []

# 改进后的选择逻辑，确保选出5个特征
for _ in range(5):  # 迭代5次，每次选择一个特征
    best_accuracy = 0
    best_feature = None
    for feature in features:
        if feature not in selected_features:
            trial_features = selected_features + [feature]
            accuracy = evaluate_feature_set(trial_features)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature
    if best_feature:
        selected_features.append(best_feature)  # 添加表现最好的特征到选定特征列表
        print(f"Added {best_feature} with accuracy {best_accuracy}")
    else:
        break  


selected_data = data[selected_features + ['RES']]
output_path = r"C:\Users\huranran\Desktop\comp\Test_model\test.py\Final_Selected_Features.csv"
data = pd.read_csv(data_path)
selected_data.to_csv(output_path, index=False)

