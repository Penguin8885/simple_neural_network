import pickle

if __name__ == '__main__':
    # 分割する対象を設定，読み込み
    #### (汎用性がない) #########
    root = './data/cifar-10-batches-py/'
    target = root + 'data_batch_5'
    with open(target, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    num_valid = 5000                # valid dataの数
    num_train = 10000 - num_valid   # train dataの数
    ############################

    # トレーニングデータを作成
    train_batch_label = ('split training batch, which is created by '
                            + data[b'batch_label'].decode('utf-8')).encode('utf-8')
    train_labels      = data[b'labels'][:num_train]
    train_data        = data[b'data'][:num_train]
    train_filename    = data[b'filenames'][:num_train]

    train = {
        b'batch_label':train_batch_label,
        b'labels':train_labels,
        b'data':train_data,
        b'filenames':train_filename
    }

    # バリデーションデータを作成
    valid_batch_label = ('validation batch, which is created by '
                            + data[b'batch_label'].decode('utf-8')).encode('utf-8')
    valid_labels      = data[b'labels'][num_train:]
    valid_data        = data[b'data'][num_train:]
    valid_filename    = data[b'filenames'][num_train:]

    valid = {
        b'batch_label':valid_batch_label,
        b'labels':valid_labels,
        b'data':valid_data,
        b'filenames':valid_filename
    }

    # 保存
    with open(target+'_kai', 'wb') as f:
        pickle.dump(train, f)
    with open(root+'/valid_batch', 'wb') as f:
        pickle.dump(valid, f)
