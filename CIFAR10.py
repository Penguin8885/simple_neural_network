import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image

import torch
import torchvision.transforms as transf

class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

class CIFAR10_Datasets():
    def __init__(self, root='./', transform=None):
        self.base_folder = 'cifar-10-batches-py'
        self.root = os.path.expanduser(root)
        self.transform = transform

        # set filename
        self.meta_filename = 'batches.meta'
        self.train_filenames = [
            'data_batch_1',
            'data_batch_2',
            'data_batch_3',
            'data_batch_4',
            'data_batch_5_kai'  # 5番目のバッチを分割・生成
        ]
        self.valid_filenames = [
            'valid_batch'       # 5番目のバッチを分割・生成
        ]
        self.test_filenames = [
            'test_batch'
        ]

        # load data
        self.label_names   = self.__load_meta(self.meta_filename)
        self.train_dataset = self.__load_data(self.train_filenames)
        self.valid_dataset = self.__load_data(self.valid_filenames)
        self.test_dataset  = self.__load_data(self.test_filenames)

    def __unpickle(self, filename):
        filename = os.path.join(self.root, self.base_folder, filename)
        print('load', filename)

        with open(filename, 'rb') as f:
            dict_ = pickle.load(f, encoding='bytes')
        return dict_

    def __load_meta(self, meta_filename):
        meta = self.__unpickle(meta_filename)
        return meta[b'label_names'] # ラベル名のみ取り出す

    def __load_data(self, data_filenames):
        data_list   = []
        labels_list = []
        for file_ in data_filenames:
            dict_ = self.__unpickle(file_)
            data_list.append(dict_[b'data'])    # データを連結(np_arrayのリスト)
            labels_list += dict_[b'labels']     # ラベルを連結
        data_list = np.concatenate(data_list)   # np_arrayとして整理
        data_list = data_list.reshape(len(data_list), 3, 32, 32).transpose(0, 2, 3, 1)
                        # データを画像の形に変換，インデックスをdata, H, W, channelの順に変換

        return CIFAR10(data_list, labels_list, self.transform)

    #### utility #########################################
    def show_img(self, img, label, img_filename=None):
        print(self.label_names[label])
        plt.imshow(img)
        if img_filename is None:
            plt.show()
        else:
            print(img_filename)
            plt.savefig(img_filename)

    def show_imgs(self, dataset):
        for i in range(len(dataset)):
            img, label = dataset[i]
            self.show_img(img, label)
    ######################################################


class CIFAR10_Loader:
    def __init__(self, root='./data'):
        self.batch_size = 4   # バッチサイズ
        self.num_workers = 4  # データロード時に使用するthreadの数
        self.transform = transf.Compose(
            [
                transf.ToTensor(),
                transf.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            ]
        )
        self.datasets = CIFAR10_Datasets(root, self.transform)

        # 注意: DataLoaderに通すと(batch_num, Channel, Height, Width)の順になる
        self.train_loader = torch.utils.data.DataLoader(
            self.datasets.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.datasets.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.datasets.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        self.classes = self.datasets.label_names



if __name__ == '__main__':
    print('show test data')
    x = CIFAR10_Datasets(root='./data')
    y = x.test_dataset
    x.show_imgs(y)
