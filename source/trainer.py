import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# reconstruction loss weight - DA SPOSTARE DA QUI
recon_weight = 0.5 # previous value: 0.8
# define GCODLoss

class GCODLoss(nn.Module):
    def __init__(self, num_classes, q=0.7, k=2, gamma=0.5, threshold=0.5, label_smoothing=0.1, eps=1e-7):
        """
        Generalized Cross-Entropy with Outlier Detection (GCOD) Loss with Label Smoothing.

        Args:
            num_classes (int): Number of classes.
            q (float): Parameter for the Generalized Cross-Entropy (GCE) part. Defaults to 0.449.
            k (int): Number of smallest confident scores to consider for outlier detection. Defaults to 2.
            gamma (float): Weighting factor for the outlier detection loss. Defaults to 0.6309.
            threshold (float): Confidence score threshold. Defaults to 0.5.
            label_smoothing (float): Label smoothing factor between 0 and 1. Defaults to 0.1.
            eps (float): Small constant for numerical stability. Defaults to 1e-7.
        """
        super(GCODLoss, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.k = k
        self.gamma = gamma
        self.threshold = threshold
        self.label_smoothing = label_smoothing
        self.eps = eps

    def forward(self, logits, targets):
        # Convert targets to one-hot encoding with label smoothing
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float().to(logits.device)
        smooth_targets = (1 - self.label_smoothing) * targets_one_hot + self.label_smoothing / self.num_classes

        # Compute softmax and clip values for numerical stability
        logits = logits.clamp(min=-15, max=15)
        softmax_output = F.softmax(logits, dim=1)
        softmax_output = torch.clamp(softmax_output, min=self.eps, max=1.0)

        # Compute the GCE loss with smoothed targets
        prod = (softmax_output * smooth_targets).sum(dim=1)
        prod = torch.clamp(prod, min=self.eps)
        gce_loss = torch.mean((1. - prod**self.q) / self.q)

        # Compute the outlier detection loss
        confident_scores = 1.0 - softmax_output
        smallest_k_confident_scores, _ = torch.topk(confident_scores, min(self.k, confident_scores.size(1)), dim=1, largest=False)
        noisy_samples = torch.sum(smallest_k_confident_scores, dim=1) < self.threshold

        # Handle case when no noisy samples are found
        outlier_loss = torch.mean(smallest_k_confident_scores[noisy_samples]) if torch.any(noisy_samples) else 0.0

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
