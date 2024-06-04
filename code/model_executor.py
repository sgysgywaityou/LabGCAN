import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from GCAN import GCN_GAT


class ModelExecutor:
    def __init__(self, model: GCN_GAT, train_size, valid_size, test_size, word_nums, label_nums, save_root) -> None:
        self.model = model
        self.save_root = save_root
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.word_nums = word_nums
        self.label_nums = label_nums
        self.doc_nums = train_size + valid_size + test_size

    def load_params(self, max_epoch, lr):
        self.max_epoch = max_epoch
        self.lr = lr
        self.optim = optim.Adam(params=self.model.parameters(), lr=lr)
        self.BCE_loss = torch.nn.BCELoss()

    def train_epoch(self, X, doc_idx_list, G, label):
        idx_list = torch.arange(start=0, end=self.word_nums+self.train_size+self.label_nums)
        idx_list[self.word_nums:self.word_nums+self.train_size] = doc_idx_list + self.word_nums # [..., 105, 100, 102, 101, 103, 104, ...]
        epoch_G = G[idx_list, :][:, idx_list]
        self.optim.zero_grad()
        pred = self.model(X, epoch_G)
        loss: torch.Tensor = self.BCE_loss(pred, label)
        loss.backward()
        self.optim.step()
        return pred, loss.item()

    def train(self, trainloader, validloader, G):
        train_losses = []
        valid_losses = [np.inf]
        for epoch in range(self.max_epoch):
            self.model.train()
            epoch_train_losses = []
            epoch_valid_losses = []
            start_time = time.time()
            for idx, (X, y, doc_idx) in enumerate(trainloader):
                print("batch, X: {}, y: {}".format(X.shape, y.shape))
                pred, loss = self.train_epoch(X, doc_idx, G, y)
                epoch_train_losses.append(loss)
                message = 'Epoch [{}/{}], loss: {:.4f}'.format(epoch+1, self.max_epoch, np.mean(epoch_train_losses))
                print(message)
            end_time = time.time()
            train_loss = np.mean(epoch_train_losses)
            train_losses.append(train_loss)
            valid_loss = self.valid(validloader)
            message = 'Epoch [{}/{}] train loss: {:.4f}, valid loss: {:.4f}, {:.1f}s'.format(epoch+1, self.max_epoch,
                                 train_loss, valid_loss, (end_time - start_time))
            print(message)
            if valid_loss < np.min(valid_losses):
                self.save_model("model_best.pt")
                print('Valid loss decrease from {:.4f} to {:.4f}, saving to {}'.format(np.min(valid_losses), valid_loss, os.path.join(self.save_path, "model_best.pt")))
            valid_losses.append(valid_loss)
        print("train over...")
        # save the model
        self.save_model("model_final.pt")
        self.draw_losses(losses=[train_losses, valid_losses], titles=["train loss", "valid loss"], mode="train and valid")
            

    def valid(self, validloader):
        self.model.eval()
        valid_losses = []
        for idx, (X, y) in enumerate(validloader):
            pred, loss = self.test_epoch(X, y)
            valid_losses.append(loss)
        valid_loss = np.mean(valid_losses)
        message = 'Evaluate complete... Valid loss: {:.4f}'.format(valid_loss)
        print(message)
        return valid_loss

    def test(self, testloader):
        self.model.eval()
        test_losses = []
        for idx, (X, y) in enumerate(testloader):
            pred, loss = self.test_epoch(X, y)
            test_losses.append(loss)
        test_loss = np.mean(test_losses)
        message = 'Test complete... Test loss: {:.4f}'.format(test_loss)
        print(message)
     
    def test_epoch(self, X, label):
        with torch.no_grad():
            pred = self.model(X)
            loss: torch.Tensor = self.BCE_loss(pred, label)
        return pred, loss.item()

    def save_model(self, model_file):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, model_file))

    def draw_losses(self, losses: list, titles: list, mode: str="train"):
        for loss, title in zip(losses, titles):
            plt.plot(loss, label=title)
        plt.legend()
        plt.title("loss curve")
        plt.savefig(os.path.join(self.save_path, f"{mode} loss.png"))
        plt.close()