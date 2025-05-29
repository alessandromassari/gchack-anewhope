import os
import torch
import logging
from typing import Tuple

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)  
    return data
    
def create_dirs():
    directories = ['checkpoints','logs']
    for directory in directories:
        os.makedirs(directory, exist_ok= True)
# LOGGING FORSE DA IMPLEMENTARE DIRETTAMENTE IN MAIN TRAMITE LOGGING LIB
#def set_logging(model, test_dir_name, ep: int)
#    filename = f"logs/model_{test_dir_name}_epoch_{ep}.pth"
#   print(f"Checkpoint saved to {filename}")

                
def save_checkpoint(model, test_dir_name: str, ep: int, val_accuracy=None):
            filename = f"checkpoints/model_{test_dir_name}_epoch_{ep}.pth"
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': ep, 
                        'val_accuracy': val_accuracy}, filename)
            print(f" >> Checkpoint saved to: {filename}")
