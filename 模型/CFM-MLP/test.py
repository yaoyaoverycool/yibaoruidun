import torch
import torch.nn as nn
import pandas as pd
from tqdm import trange
from torch.utils.data import DataLoader
from model2 import IdentifyModel
from dataset import IdentifyDataset
import torch.nn.functional as F

DEVICE = "cpu"
MODEL_LOAD = None
TEST = False
MODEL_LOAD = "./model/model_final.pth"
TEST = True
EPOCH = 100
BATCH_SIZE = 32
lr = 0.0000625
eps=1.5e-4
test_frequence = EPOCH // 10

TrainexcelFile = "trainData_SMOTE.xlsx"
TestexcelFile = "testData.xlsx"
TrainData = pd.read_excel(TrainexcelFile)
TestData = pd.read_excel(TestexcelFile)

model = IdentifyModel().to(DEVICE)
if MODEL_LOAD is not None:
    model.load_state_dict(torch.load(MODEL_LOAD))
lossFun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr, eps=eps, weight_decay=0.0001)
data1 = IdentifyDataset(TrainData)
trainDataloader = DataLoader(data1, BATCH_SIZE, shuffle=True)
data2 = IdentifyDataset(TestData)
testDataloader = DataLoader(data2, BATCH_SIZE, shuffle=True)
train_len = len(data1)
test_len = len(data2)

def test(model:IdentifyModel):
    model.eval()
    totalAccuracy0 = 0
    totalAccuracy1 = 0
    total_tp = 0  # 总体真正例  
    total_fp = 0  # 总体假正例  
    total_fn = 0  # 总体假负例  
    total_tn = 0  # 总体真负例  
    total_len0 = 0
    total_len1 = 0
    total_len = 0
    totalLoss = 0
    with torch.no_grad():
        for data in testDataloader:
            x, y = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            outputs = model(x)
            loss = lossFun(outputs, y)
            totalLoss += loss
            # 计算每个类别的预测结果  
            y_pred = torch.argmax(outputs, dim=1)  
            # 计算TP, FP, FN, TN  
            tp = ((y_pred == 1) & (y == 1)).sum().item()
            fp = ((y_pred == 1) & (y == 0)).sum().item()  
            fn = ((y_pred == 0) & (y == 1)).sum().item()  
            tn = ((y_pred == 0) & (y == 0)).sum().item() 
            
            # 累加  
            total_tp += tp  
            total_fp += fp  
            total_fn += fn  
            total_tn += tn  
            accuracy0 = (torch.argmax(outputs, dim=1)[y==0] == y[y==0]).sum()
            accuracy1 = (torch.argmax(outputs, dim=1)[y==1] == y[y==1]).sum()
            totalAccuracy0 += accuracy0
            totalAccuracy1 += accuracy1
            # 累加类别长度  
            total_len0 += len(y[y==0])
            total_len1 += len(y[y==1])
            total_len += len(y)
            
    acc0 = totalAccuracy0 / total_len0
    acc1 = totalAccuracy1 / total_len1
    
    totalLoss /= test_len
    return totalLoss, acc0, acc1

def train(model:IdentifyModel):
    tr = trange(1, int(EPOCH) + 1)
    for epoch in tr:
        model.train()
        totalAccuracy0 = 0
        totalAccuracy1 = 0
        totalAccuracy = 0
        total_len0 = 0
        total_len1 = 0
        total_len = 0
        totalLoss = 0
        for data in trainDataloader:
            x, y = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            outputs = model(x)
            y_one_hot = F.one_hot(y,num_classes=2).float()
            loss = lossFun(outputs, y_one_hot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss
            accuracy0 = (torch.argmax(outputs, dim=1)[y==0] == y[y==0]).sum()
            accuracy1 = (torch.argmax(outputs, dim=1)[y==1] == y[y==1]).sum()
            accuracy = (torch.argmax(outputs, dim=1) == y).sum()
            totalAccuracy0 += accuracy0
            totalAccuracy1 += accuracy1
            totalAccuracy += accuracy
            total_len0 += len(y[y==0])
            total_len1 += len(y[y==1])
            total_len += len(y)
        acc0 = totalAccuracy0 / total_len0
        acc1 = totalAccuracy1 / total_len1
        acc = totalAccuracy / total_len
        totalLoss /= train_len
        tr.set_postfix_str(f"Loss:{totalLoss:.3f} Acc0:{acc0:.3f} Acc1:{acc1:.3f}")
        if (epoch % test_frequence) == 0:
            totalLoss, acc0, acc1 = test(model)
            print(f"{epoch}测试集loss:{totalLoss}")
            print(f"{epoch}测试集accuracy0:{acc0}")
            print(f"{epoch}测试集accuracy1:{acc1}")
            print(f"{epoch}测试集accuracy:{acc}")
    torch.save(model.state_dict(), f"model/model_final_{(acc0 + acc1) / 2:.2f}.pth")

if not TEST:
    train(model)
totalLoss, acc0, acc1= test(model)
 
print(f"最终测试集loss:{totalLoss}")
print(f"最终测试集accuracy0:{acc0}")
print(f"最终测试集accuracy1:{acc1}")

