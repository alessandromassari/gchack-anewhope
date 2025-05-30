import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# reconstruction loss weight - DA SPOSTARE DA QUI
recon_weight = 0.5 # previous value: 0.8
# define GCODLoss
class GCODLoss(nn.Module):
    def __init__(self, num_classes, q = 0.5, k=2, gamma=0.7, threshold = 0.5):
        super(GCODLoss, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.k = k
        self.gamma = gamma
        self.threshold = threshold

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output before softmax (batch_size, num_classes).
            targets (torch.Tensor): True labels (batch_size).
        Returns:
            torch.Tensor: The computed GCOD loss.
        """
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float().to(logits.device)

        # Compute the GCE loss
        softmax_output = F.softmax(logits, dim=1)
        gce_loss = torch.mean((1. - (softmax_output * targets_one_hot).sum(dim=1)**self.q) / self.q)

        # Compute the outlier detection loss
        # Find the k smallest confident scores
        # 1 - softmax_output is the confidence score
        confident_scores = 1.0 - softmax_output
        smallest_k_confident_scores, _ = torch.topk(confident_scores, self.k, dim=1, largest=False)
        noisy_samples = torch.sum(smallest_k_confident_scores, dim=1) < self.threshold
        outlier_loss = torch.mean(smallest_k_confident_scores[noisy_samples])

        # Combine GCE and outlier detection loss
        total_gcod_loss = gce_loss + self.gamma * outlier_loss

        return total_gcod_loss
        
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

def train_epoch(model, loader, optimizer, device, num_classes):
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    # focal loss (optional)
    gcod_loss = GCODLoss(num_classes=num_classes)
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        y = data.y.view(-1)
        #loss = F.cross_entropy(out, y)
        loss = gcod_loss(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip gradients
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_examples += y.size(0)

    return total_loss / len(loader), total_correct / total_examples

#@torch.no_grad()
        # accumulate total losses
        #total_loss += loss.item() #* data.num_graphs if hasattr(data, 'num_graphs') else loss.item() # provare weight per pesare 
        
    #return total_loss/len(td_loader)
