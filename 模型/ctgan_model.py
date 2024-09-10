# Import libraries
import torch
import pandas as pd
import seaborn as sns
import torch.nn as nn

from ctgan import CTGAN
from ctgan.synthesizers.ctgan import Generator
# Import training Data
data = pd.read_csv("./FirstCleaned_Dataset.csv")

top_5_features = ['就诊次数_SUM','月统筹金额_MAX','ALL_SUM','月药品金额_AVG','可用账户报销金额_SUM','RES']
#data = data[data['RES']==1]
print(data)
data_selected = data[top_5_features]
categorical_features = ['RES' ]
continuous_cols = ['月统筹金额_MAX', 'ALL_SUM', '月药品金额_AVG', '可用账户报销金额_SUM','就诊次数_SUM']
# Train Model
from ctgan import CTGAN
#ctgan = torch.load('D:/mod/ctgan_res.pth')
ctgan = CTGAN(embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256), batch_size=500, verbose=True, epochs=300)
ctgan.fit(data_selected, categorical_features, epochs=1000)
#torch.save(ctgan,'D:/mod/ctgan_res.pth')

# Generate synthetic_data
synthetic_data = ctgan.sample(100)
print(synthetic_data[['就诊次数_SUM','ALL_SUM']])


# Analyze Synthetic Data
from table_evaluator import TableEvaluator

print(data_selected.shape, synthetic_data.shape)
table_evaluator = TableEvaluator(data_selected, synthetic_data, cat_cols=categorical_features)
table_evaluator.visual_evaluation()
# compute the correlation matrix
corr = synthetic_data.corr()

# plot the heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")

# show summary statistics SYNTHETIC DATA
summary = synthetic_data.describe()
print(summary)
