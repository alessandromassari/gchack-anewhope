import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# reconstruction loss weight - DA SPOSTARE DA QUI
recon_weight = 0.5 # previous value: 0.8

# our beloved Kullback-Leibler term loss
def kl_loss(mu, logvar):
    # clip logvar to avoid extreme values 
    clip_logvar = torch.clamp(logvar, min=-5.0, max=5.0) 
    return -0.5 * torch.mean(1 + clip_logvar -mu.pow(2) - clip_logvar.exp())

def pretraining(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        # optionally skip pretrain: here solo supervised on classification head
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        # dummy: no pretraining in this setup
        loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    # focal loss (optional)
    # criterion = FocalLoss(gamma=2.0).to(device)
    criterion = CrossEntropyLoss()  # + label_smoothing if desideri
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        y = data.y.view(-1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_examples += y.size(0)

    return total_loss / len(loader), total_correct / total_examples

@torch.no_grad()
        # accumulate total losses
        #total_loss += loss.item() #* data.num_graphs if hasattr(data, 'num_graphs') else loss.item() # provare weight per pesare 
        
    #return total_loss/len(td_loader)
