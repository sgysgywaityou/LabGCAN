import numpy as np
from mxnet.contrib import text
import torch.utils.data as data_utils
import torch
import os
 

def load_data(data_root, batch_size=None):
    X_train, y_train = np.load(os.path.join(data_root, 'X_train.npy')), np.load(os.path.join(data_root, 'y_train.npy'))
    X_valid, y_valid = np.load(os.path.join(data_root, 'X_valid.npy')), np.load(os.path.join(data_root, 'y_valid.npy'))
    X_test, y_test = np.load(os.path.join(data_root, 'X_test.npy')), np.load(os.path.join(data_root, 'y_test.npy'))
    label_embed = np.load(os.path.join(data_root, 'label_embed.npy'))
    embed = text.embedding.CustomEmbedding('./dataset/aapd/word_embed.txt')
    # print(X_train.shape, X_valid.shape, X_test.shape, embed.idx_to_vec.asnumpy().shape)
 
    test_data = data_utils.TensorDataset(torch.from_numpy(X_test).type(torch.LongTensor),
                                         torch.from_numpy(y_test).type(torch.LongTensor),
                                         torch.arange(len(X_test)))
    train_data = data_utils.TensorDataset(torch.from_numpy(X_train).type(torch.LongTensor),
                                          torch.from_numpy(y_train).type(torch.LongTensor),
                                          torch.arange(len(X_train)))
    valid_data = data_utils.TensorDataset(torch.from_numpy(X_valid).type(torch.LongTensor),
                                          torch.from_numpy(y_valid).type(torch.LongTensor),
                                          torch.arange(len(X_valid)))
    if batch_size == None:
        train_loader = data_utils.DataLoader(train_data, batch_size=train_data.__len__(), shuffle=True, drop_last=True)
        test_loader = data_utils.DataLoader(test_data, batch_size=test_data.__len__(), drop_last=True)
        valid_loader = data_utils.DataLoader(valid_data, batch_size=valid_data.__len__(), drop_last=True)
    else:
        train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = data_utils.DataLoader(test_data, batch_size=batch_size, drop_last=True)
        valid_loader = data_utils.DataLoader(valid_data, batch_size=batch_size, drop_last=True)
    return train_loader, valid_loader, test_loader, label_embed, embed.idx_to_vec.asnumpy()


