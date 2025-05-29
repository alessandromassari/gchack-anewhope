import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from loadData import GraphDataset
import pandas as pd 
from trainer import pretraining, train_epoch
from utilities import create_dirs, save_checkpoint
from my_model import NHNmodel
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from collections import Counter

def evaluate(model, data_loader, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0.0
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                classification_loss = F.cross_entropy(output, data.y)
                total_loss += classification_loss.item()
                
    if calculate_accuracy:
        if total > 0 and len(data_loader) > 0:
            accuracy = correct / total 
            avg_loss = total_loss / len(data_loader) 
        else: 
            accuracy = 0.0
            avg_loss = 0.0
        return accuracy, avg_loss
    return predictions

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    
    # create directories
    create_dirs()
    
    # Hyperparameters for the model (circa a ctrl+c - ctrl+v from competiton GitHub)
    node_in_dim = 1
    edge_in_dim = 7
    hidden_dim = 64         # 128
    num_layers = 4          # previous val: 
    mlp_dims = (256,128)
    out_classes = 6
    dropout = 0.4
    batch_size = 16
    learning_rate = 0.0001
    weight_decay     = 1e-4
    pretrain_epochs = 0
    num_epochs = 80
    
    torch.manual_seed(0)

    # early stopping parameters (MAYBE IN THE FUTURE)
    
    # Initialize the model and choose the optimizer
    model = NHNmodel(node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        mlp_classifier_dims=mlp_dims,
        out_classes=out_classes,
        dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #node_feat_transf = gen_node_features(feat_dim = node_in_dim)

    # checkpoints saving threshold on training loss - if have time implement this on acc or validation
    model_loss_min = float('inf')

    # TO BE IMPLEMENTED FOR LOGS AT LEAST 10
    logs_counter = 0
    
    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

     # If train_path is provided then train on it
    if args.train_path:
        print(f">> Starting the train of the model using the following train set: {args.train_path}")
        all_train_dataset = GraphDataset(args.train_path)
        all_train_index = list(range(len(all_train_dataset)))
        train_index, val_index = train_test_split(all_train_index,test_size=0.2,random_state=42)
        
        # split the training set in training and validation set
        train_dataset = torch.utils.data.Subset(all_train_dataset, train_index)
        val_dataset = torch.utils.data.Subset(all_train_dataset, val_index)
        
        # classes_w definition
        classes_w = None
        if isinstance(train_dataset[0].y, torch.Tensor) and train_dataset[0].y.numel() ==1:
            train_labels = [d.y.item() for d in train_dataset]
        else:
            train_labels = [d.y for d in train_dataset]
        class_counter = Counter(train_labels)
        num_classes = len(class_counter)
        
        total_samples = len(train_labels)
        classes_w_list = [total_samples / (class_counter[i] for i in sorted(class_counter.keys()))]
        classes_w = torch.tensor(class_w_list, dtype=torch.float).to(device)
        print(f"Classes weights: {classes_w}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # ----------- pre-training loop ------------ #
        print("\n--- Starting Pre-training of the model ---")
        for epoch in range(pretrain_epochs):
            loss = pretraining(model,train_loader, optimizer, device)
            print(f"PRETRAINING: Epoch {epoch + 1}/{pretrain_epochs}, Loss: {train_loss:.4f}")
        print(f"--- Pre-training Completed ---")
        
        # -----------   Training loop   ------------ #
        best_val_acc = 0.0
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model,train_loader, optimizer, device, classes_w)
            print(f"TRAININIG: Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            
            # validation valutation every 5 epoches
            vaL_loss = 0.0
            val_accuracy = 0.0
            # evaluate on validation set every 5 epoches
            if (epoch+1) % 5 == 0 or epoch == num_epochs - 1:
                val_accuracy, val_loss = evaluate(model, val_loader, device, calculate_accuracy=True)
                print(f"VALIDATION: Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
                    save_checkpoint(model, test_dir_name, epoch, val_accuracy) 
                    print(f"Checkpoint saved for epoch {epoch + 1} with validation accuracy {val_accuracy:.4f}")
        print(f"--- Training Completed ---")
            
    # Else if train_path NOT provided 
    if not args.train_path:
        checkpoint_path = args.checkpoint
        # raise an error if not able to find the checkpoint model
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found! {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f">> Loading pre-training model from: {checkpoint_path}")
          
    # Evaluate and save test predictions
    predictions = evaluate(model, test_loader, device, calculate_accuracy=False)
    test_graph_ids = list(range(len(predictions)))  # Generate IDs for graphs

    # Save predictions to CSV
    test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]
    #output_csv_path = os.path.join(f"testset_{test_dir_name}.csv")
    output_csv_path = os.path.join('/kaggle/working/', f"testset_{test_dir_name}.csv")
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")

# arguments plus call to the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a classification model on graph datasets.")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint model (e.g. checkpoints/model_B_epoch_10.pth)")
    args = parser.parse_args()
    main(args)
