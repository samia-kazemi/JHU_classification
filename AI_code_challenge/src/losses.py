import torch.nn as nn

class MosLoss(nn.Module):
    def __init__(self):
        super(MosLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.loss_fn(input, target)
