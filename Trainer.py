import os
import datetime
import re

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

def str_now():
    return datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

class Trainer:
    def __init__(self, device, net, criterion, dataset_loader):
        self.device = device
        self.net = net.to(self.device)
        self.criterion = criterion
        self.dataset_loader = dataset_loader
        self.train_loader = self.dataset_loader.train_loader
        self.valid_loader = self.dataset_loader.valid_loader
        self.test_loader = self.dataset_loader.test_loader

        print('\n@ Trainer >> constitution information')
        print('net:', self.net)
        print('criterion:', self.criterion)
        print('\n@ Trainer >> dataset information')
        print('batch_size:', self.dataset_loader.batch_size)
        print('train_data_batch_num:', len(self.train_loader))
        print('valid_data_batch_num:', len(self.valid_loader))
        print('test_data_batch_num:', len(self.test_loader))
        print('transform:', self.dataset_loader.transform)
        print('num_workers:', self.dataset_loader.num_workers)
        print('\n@ Trainer >> device information')
        print('device:', self.device)

    def train(self, init_lr, min_lr, init_params=None):
        print('\n%%%%%%%%%%%%%%%% TRAINING INFORMATION %%%%%%%%%%%%%%%%')
        print('@ Trainer >> training information')
        print('initial learning rate:', init_lr)
        print('minimal learning rate:', min_lr)
        print('initial parameters:', init_params)

        # 初期化
        if init_params is not None:
            self.net.load_state_dict(torch.load(init_params))
            epoch = int(re.sub(r'\D', '', init_params)) + 1 # エポック番号の取り出し
        else:
            epoch = 1
        lr = init_lr
        train_loss = 0
        old_valid_loss = np.inf
        optimizer = optim.SGD(self.net.parameters(), lr=lr) # optimizerはSGD固定

        # 繰り返し学習
        while True:
            print('\n======== epoch', epoch, 'training start ========')
            print(str_now(), '\t learning rate:', lr)

            loss_print_n = int(len(self.train_loader) / 5)  # 途中経過を表示するバッチ数
            for i, data in enumerate(self.train_loader):
                inputs, answers = data
                inputs, answers = inputs.to(self.device), answers.to(self.device)
                                    # 設定されたデバイス(GPU or CPU)のメモリへ転送
                outputs = self.net(inputs)
                loss = self.criterion(outputs, answers)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % loss_print_n == (loss_print_n-1):
                    print(str_now(), '\t',
                        'batch: %5d\tloss: %.3f' % (i+1, train_loss/loss_print_n))
                    train_loss = 0

            else:
                print('======== epoch', epoch, 'training finished ========')
                new_valid_loss = self.test(self.valid_loader)    # valid_dataでtest, ロスを計算
                print('learning rate:', lr)
                print('new loss:', new_valid_loss, end='')
                print(('   <O   ' if new_valid_loss < old_valid_loss else '   >X   '), end='')
                print('old loss:', old_valid_loss)

                # ロスが小さくなったかを確認
                if new_valid_loss < old_valid_loss:
                    print('saving the parameters')
                    torch.save(self.net.state_dict(), 'params'+str(epoch)+'.pth')
                    epoch += 1
                    old_valid_loss = new_valid_loss
                    print('continue with same learning rate')

                # ロスが小さくなっていない場合はlearning rateを下げる
                else:
                    lr /= 2.0
                    if lr >= min_lr:
                        print('loading the last parameters')
                        self.net.load_state_dict(torch.load('params'+str(epoch-1)+'.pth'))
                        optimizer = optim.SGD(self.net.parameters(), lr=lr) # optimizerはSGD固定
                        print('continue with new learing rate; lr =', lr)
                    else:
                        print('next_lr =', lr, ' < min_lr =', min_lr)
                        print('the next learning rate is less than threshold')
                        print('finished all training')
                        break

            # print('[Option] rm -r pymp-*: ' + ('success' if os.system('rm -r pymp-*') == 0 else 'failure'))
            #             # 通常は不要，サーバーがビジーで一時ファイルを消せないときのために使用

    def test(self, test_loader):
        print('################ test start ################')
        correct = 0; total_datasize = 0
        test_loss = 0; total_loss = 0
        with torch.no_grad():
            loss_print_n = int(len(test_loader) / 5)  # 途中経過を表示するバッチ数
            for i, data in enumerate(test_loader):
                inputs, answers = data
                inputs, answers = inputs.to(self.device), answers.to(self.device)
                                    # 設定されたデバイス(GPU or CPU)のメモリへ転送
                outputs = self.net(inputs)

                #### lossの計算 ####
                loss = self.criterion(outputs, answers).item()
                test_loss += loss
                total_loss += loss

                #### accuracyの計算 ####
                _, predicted = torch.max(outputs.data, dim=1)  # 1バッチのデータすべての予測値を取り出す
                correct += (predicted == answers).sum().item() # 1バッチのデータで正解であったものを数える
                total_datasize += answers.size(0)

                if i % loss_print_n == (loss_print_n - 1):
                    print(str_now(), '\t',
                        'batch: %5d\tloss: %.3f' % (i+1, test_loss/loss_print_n))
                    test_loss = 0
        print('################ test finished ################')
        print('Accuracy:', correct/total_datasize)
        return total_loss

    def final_test(self):
        print('\n%%%%%%%%%%%%%%%% FINAL TEST INFORMATION %%%%%%%%%%%%%%%%')
        self.test(self.test_loader)