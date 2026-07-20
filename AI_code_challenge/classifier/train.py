import json
import os
from AI_code_challenge.src.dataset import MosDataset
from AI_code_challenge.src.models import ResNet18
from AI_code_challenge.src.losses import MosLoss
from AI_code_challenge.src.trainer import Trainer
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

def Train(config_file):

    # Load configuration as dictionary
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create experiment folder and save config file
    os.makedirs(Path('experiments') / config['exp_name'], exist_ok=True)
    with open(Path('experiments') / config['exp_name'] / 'config.json', 'w') as f:
        json.dump(config, f)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    # Get datasets
    train_dataset = MosDataset(config, split = "Train")
    valid_dataset = MosDataset(config, split = "Valid")

    # Get dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"],
                                  shuffle=True, num_workers=config["num_workers"],
                                  pin_memory=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config["batch_size"],
                                  shuffle=False, num_workers=config["num_workers"],
                                  pin_memory=True)

    # Get model
    model_name_to_class = {"ResNet18": ResNet18}
    model = model_name_to_class[config["model_name"]](**config["model_kwargs"])
    model.to(device)

    # Get loss
    loss_fn = MosLoss().to(device)

    # Get optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Define trainer
    trainer = Trainer(model, optimizer, loss_fn, device, Path("experiments") / config["exp_name"])

    # Train model
    trainer.train(train_dataloader, valid_dataloader, config["max_epochs"])


