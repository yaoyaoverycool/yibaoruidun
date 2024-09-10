from torch import nn

class CFM_MLP(nn.Module):
    def __init__(self):  
        super(CFM_MLP, self).__init__()  
          
        # 定义模型的各层  
        self.fc1 = nn.Linear(5, 128)  
        self.bn1 = nn.BatchNorm1d(128)  
        self.dr1 = nn.Dropout()  
        self.fc2 = nn.Linear(128, 256)  
        self.bn2 = nn.BatchNorm1d(256)  
        self.dr2 = nn.Dropout()  
        self.fc3 = nn.Linear(256, 128)  
        self.bn3 = nn.BatchNorm1d(128)  
        self.dr3 = nn.Dropout()  
        self.fc4 = nn.Linear(128, 64)  
        self.bn4 = nn.BatchNorm1d(64)  
        self.dr4 = nn.Dropout()  
        self.fc5 = nn.Linear(64, 2)  
          
        # 定义模型层序列  
        self.layers = nn.Sequential(  
            self.fc1,  
            self.bn1,  
            nn.LeakyReLU(),  
            self.dr1,  
            self.fc2,  
            self.bn2,  
            nn.LeakyReLU(),  
            self.dr2,  
            self.fc3,  
            self.bn3,  
            nn.LeakyReLU(),  
            self.dr3,  
            self.fc4,  
            self.bn4,  
            nn.LeakyReLU(),  
            self.dr4,  
            self.fc5,  
        )

    def forward(self, x):
        return self.layers(x)