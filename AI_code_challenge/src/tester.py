import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

class Tester(nn.Module):
    def __init__(self, model, loss_fn, device, exp_dir, species_names):
        super().__init__()
        self.model = model
        self.loss = loss_fn
        self.device = device
        self.exp_dir = exp_dir
        self.species_names = species_names
    
    def test(self, test_dataloader):
        total_loss = 0
        y = []
        preds = []
        self.model.eval()
        for X_batch, y_batch in tqdm(test_dataloader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            with torch.no_grad():
                y_pred = self.model(X_batch)
                y.append(y_batch.view(-1, ).detach().cpu().numpy())
                preds.append(y_pred.detach().cpu().numpy())
                loss = self.loss(y_pred, y_batch)
                
            total_loss += loss.item()
        
        test_loss = total_loss / len(test_dataloader.dataset)
        y = np.concatenate(y, axis = 0)
        preds = np.concatenate(preds, axis = 0)
        preds = np.argmax(preds, axis = -1)
        acc = accuracy_score(y, preds)
        #cm = confusion_matrix(y, preds, labels = self.species_names, normalize = 'true')
        cm = confusion_matrix(y, preds, normalize = 'true')

        return (test_loss, y, preds, acc, cm)
