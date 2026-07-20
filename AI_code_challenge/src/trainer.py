import os
import torch.nn as nn
import torch
from tqdm import tqdm


class Trainer(nn.Module):
    def __init__(self, model, optimizer, loss_fn, device, exp_dir):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss_fn
        self.device = device
        self.exp_dir = exp_dir
    
    def train(self, train_dataloader, valid_dataloader, epochs):
        best_val_loss = float('inf')
        for epoch in range(epochs):

            print("Training epoch: ")
            loss_train = self.train_epoch(train_dataloader, mode = 'train')
            print("Validating epoch: ")
            loss_valid = self.train_epoch(valid_dataloader, mode = 'valid')

            print(epoch, loss_train, loss_valid)

            if  best_val_loss > loss_valid:
                print(f"Saving model at epoch **** {epoch} *****")
                best_val_loss = loss_valid
                torch.save(self.model.state_dict(), os.path.join(self.exp_dir, "best_loss.pth"))
    
    def train_epoch(self, dataloader, mode = 'train'):
        
        total_loss = 0
        #print("dataloader = ", dataloader)
        for X_batch, y_batch in tqdm(dataloader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            if mode == 'train':
                self.model.train()

                y_pred = self.model(X_batch)
                loss = self.loss(y_pred, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            else:
                self.model.eval()
                with torch.no_grad():
                    y_pred = self.model(X_batch)
                    loss = self.loss(y_pred, y_batch)

            total_loss += loss.item()
        
        return total_loss / len(dataloader.dataset)
