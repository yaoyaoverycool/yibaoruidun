from torch import nn

class IdentifyModel(nn.Module):
    def __init__(self):
        super(IdentifyModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(5, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(64, 2),
            # nn.LeakyReLU(),
        )
        # self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.linear(x)