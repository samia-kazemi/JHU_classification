import json
import os
from AI_code_challenge.src.dataset import MosDataset
from AI_code_challenge.src.models import ResNet18
from AI_code_challenge.src.losses import MosLoss
from AI_code_challenge.src.tester import Tester
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

def Test(config_file):

    # Load configuration as dictionary
    with open(config_file, 'r') as f:
        org_config = json.load(f)
    
    # Load configuration file from the experiment folder
    with open(Path('experiments') / org_config['exp_name'] / 'config.json', 'r') as f:
        config = json.load(f)
        config["mode"] = "Test"
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    # Get datasets
    test_dataset = MosDataset(config, split = "Test")

    # Get dataloader
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"],
                                 shuffle=False, num_workers=config["num_workers"],
                                 pin_memory=True)

    # Get model
    model_name_to_class = {"ResNet18": ResNet18}
    model = model_name_to_class[config["model_name"]](**config["model_kwargs"])
    # Load model parameters
    model.load_state_dict(torch.load(Path("experiments") / config["exp_name"] / "best_loss.pth"))
    model.to(device)

    # Get loss
    loss_fn = MosLoss().to(device)

    # Define tester
    tester = Tester(model, loss_fn, device, Path("experiments") / config["exp_name"], test_dataset.species_names)

    # Test model
    test_loss, y, preds, acc, cm = tester.test(test_dataloader)
    print("Test loss = ", test_loss)
    print("Accuracy = ", acc)
    print("Confusion matrix = ", cm)

