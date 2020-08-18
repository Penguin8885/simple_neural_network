import sys
import torch

import Net
import Loss
import CIFAR10
import Trainer

def set_seed(random_seed):
    print('random_seed:', random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def get_device_name(device):
    if device == 'gpu':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        return 'cpu'

if __name__ == '__main__':
    # セッティング
    set_seed(777)                       # 乱数シードを設定

    device = get_device_name('gpu')     # デバイス名(cpuかgpuか)を取得
    net = Net.SixFullyConnectedNet()    # 6層の全結合モデルを作成
    dataset_loader = CIFAR10.CIFAR10_Loader('./data') # CIFER10のデータセットを読み込み
    criterion = Loss.CrossEntropyLoss() # クロスエントロピー損失を読み込み

    # トレーニングの設定
    trainer = Trainer.Trainer(
        device=device,                  # 計算デバイスを設定
        net=net,                        # ニューラルネット構成を設定
        criterion=criterion,            # 損失関数を設定
        dataset_loader=dataset_loader   # データセットを設定
    )

    # トレーニング
    trainer.train(
        init_lr=0.05,       # トレーニングのステップサイズの初期値
        min_lr=0.0001,      # トレーニングのステップサイズの最小値
        init_params=None    # 初期パラメータ(パラメータ転移などを行う場合)
    )

    # テスト
    trainer.final_test()
