import torch
import torch.nn as nn
from data import load_data
from model_executor import ModelExecutor


if __name__ == '__main__':
    data_root = ''
    save_root = ''
    train_loader, valid_loader, test_loader, label_embed, embed = load_data(data_root=data_root)
    embed_dim = 300
    input_len = 500
    train_size = 53840
    valid_size = 1000
    test_size = 1000
    word_nums = 69397
    label_nums = 54
    hidden_dim = 64
    graph_attn_dim = 64
    attn_dim = 64
    k = 8
    co_k = 8
    gcn_dim = 128
    out_dim = 32
    alpha = 0.5
    beta = 0.5
    eta = 0.7
    gamma = 0.5
    co_attn_dim = 64
    lstm_layers = 2
    model = CoreModel(input_len=input_len, label_nums=label_nums, input_dim=embed_dim, hidden_dim=hidden_dim, graph_attn_dim=graph_attn_dim, attn_dim=attn_dim, k=k, co_k=co_k, gcn_dim=gcn_dim, out_dim=out_dim, embeddings=embed, alpha=alpha, beta=beta, eta=eta, gamma=gamma, co_attn_dim=co_attn_dim, lstm_layers=lstm_layers)
    executor = ModelExecutor(model, train_size, valid_size, test_size, word_nums, label_nums, save_root=save_root)
    lr = 1e-4
    max_epoch = 100
    executor.load_params(max_epoch=max_epoch, lr=lr)
    executor.train(trainloader=train_loader, validloader=valid_loader)
    executor.test(testloader=test_loader)
    print("Over...")