import torch.nn as nn
import torch.nn.functional as F

class SixFullyConnectedNet(nn.Module):
    def __init__(self):
        super(SixFullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(3072, 2048)    # 32*32*3入力, 2048出力
        self.fc2 = nn.Linear(2048, 2048)    # 2048入力, 2048出力
        self.fc3 = nn.Linear(2048, 2048)    # 2048入力, 2048出力
        self.fc4 = nn.Linear(2048, 2048)    # 2048入力, 2048出力
        self.fc5 = nn.Linear(2048, 2048)    # 2048入力, 2048出力
        self.fc6 = nn.Linear(2048, 10)      # 2048入力, 10出力


    def forward(self, x):
        # 画像をベクトル化
        size = x.size()
        x = x.view(-1, size[1]*size[2]*size[3])  #  reshape(1, チャンネル数*縦*横)

        # 作用
        x = F.sigmoid(self.fc1(x))  # 全結合 -> シグモイド活性化
        x = F.sigmoid(self.fc2(x))  # 全結合 -> シグモイド活性化
        x = F.sigmoid(self.fc3(x))  # 全結合 -> シグモイド活性化
        x = F.sigmoid(self.fc4(x))  # 全結合 -> シグモイド活性化
        x = F.sigmoid(self.fc5(x))  # 全結合 -> シグモイド活性化
        x = self.fc6(x)             # 全結合 -> 活性化なし(損失関数の設定による)

        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(3,6,5)       # 1入力,  6出力, 5*5の畳み込み
        self.conv2 = nn.Conv2d(6,16,5)      # 6入力, 16出力, 5*5の畳み込み
        self.pool = nn.MaxPool2d(2,2)       # Maxプーリング，サイズ(2,2)
        self.fc1 = nn.Linear(16*5*5, 120)   # 16*5*5入力, 120出力
        self.fc2 = nn.Linear(120, 84)       # 120入力, 84出力
        self.fc3 = nn.Linear(84, 10)        # 84入力, 10出力

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 畳み込み      -> ReLU活性化
        x = self.pool(x)          # Maxプーリング -> 活性化なし
        x = F.relu(self.conv2(x)) # 畳み込み      -> ReLU活性化
        x = self.pool(x)          # Maxプーリング -> 活性化なし

        size = x.size()
        x = x.view(-1, size[1]*size[2]*size[3])  #  reshape(1, チャンネル数*縦*横)

        x = F.relu(self.fc1(x))   # アフィン変換 -> ReLU活性化
        x = F.relu(self.fc2(x))   # アフィン変換 -> ReLU活性化
        x = self.fc3(x)           # アフィン変換 -> 活性化なし

        return x
