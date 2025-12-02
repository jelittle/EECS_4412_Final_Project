
"""

paper:Deep Pyramid Convolutional Neural Networks for Text Categorization
https://riejohnson.com/paper/dpcnn-acl17.pdf

from:
https://github.com/Cheneng/DPCNN/blob/master/model/DPCNN.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader



class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, config):
        super(DPCNN, self).__init__()
        self.config = config
        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, self.config.word_embedding_dimension), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(self.channel_size, 2)  # Changed from 2*channel_size to channel_size

    def forward(self, x):
        batch = x.shape[0]

        # Region embedding
        x = self.conv_region_embedding(x)        # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)                      # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            x = self._block(x)

        # Global max pooling over remaining sequence dimension
        x = torch.max(x, dim=2)[0]  # [batch_size, channel_size, 1]
        x = x.view(batch, -1)  # Flatten
        x = self.linear_out(x)

        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels
    
class Config:
    word_embedding_dimension = 10   
    num_classes = 3 
def pad_embeddings(embeddings, max_len, vector_size):
    """Pad or truncate embeddings to max_len."""
    padded = []
    for emb in embeddings:
        emb = np.array(emb)
        if len(emb) == 0:
            padded.append(np.zeros((max_len, vector_size)))
        elif emb.ndim == 1:
            emb = emb.reshape(1, -1)
            padded.append(np.vstack([emb, np.zeros((max_len - 1, vector_size))]) if len(emb) < max_len else emb[:max_len])
        else:
            if len(emb) > max_len:
                padded.append(emb[:max_len])
            elif len(emb) < max_len:
                padded.append(np.vstack([emb, np.zeros((max_len - len(emb), vector_size))]))
            else:
                padded.append(emb)
    return np.stack(padded)

def load_and_prepare_data(data_path, max_samples=None):
    """Load data and prepare for training."""
    print("Loading data...")
    loaded = np.load(data_path, allow_pickle=True)
    data = loaded["data"][:max_samples] if max_samples else loaded["data"]
    
    targets, embeddings = data[:, 0], data[:, 1]
    
    # Calculate padding params
    lengths = [len(e) for e in embeddings]
    max_len = int(np.percentile(lengths, 95))
    vector_size = np.array(embeddings[0]).shape[-1] if np.array(embeddings[0]).ndim > 1 else len(embeddings[0])
    
    print(f"Seq length: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}, using max_len={max_len}")
    
    # Pad and encode
    X = pad_embeddings(embeddings, max_len, vector_size)
    y = np.array([{'NEG': 0, 'NEU': 1, 'POS': 2}[label] for label in targets])
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    return X, y

if __name__ == '__main__':
    # Load and prepare data
    X, y = load_and_prepare_data("../project_data/t4sa_data_w2v_sg_dpcnn.npz", max_samples=10000)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)  # (batch, 1, seq_len, embed_dim)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_t = torch.LongTensor(y_test)

    print(f"Train tensor shape: {X_train_t.shape}")  # Should be (n, 1, seq_len, 10)

    # Create DataLoader
    batch_size = 64
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    config = Config()
    model = DPCNN(config)
    
    # Fix output layer for 3 classes (matches the global max pooled output)
    model.linear_out = nn.Linear(model.channel_size, config.num_classes)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"\nTraining on {device}")
    print(f"Model: DPCNN with {config.word_embedding_dimension}-dim embeddings, {config.num_classes} classes")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}\n")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    # Final evaluation
    from sklearn.metrics import classification_report, confusion_matrix
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    # Classification report
    class_names = ['NEG', 'NEU', 'POS']
    print("\n" + "="*50)
    print("Final Classification Report:")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    
