import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.utils.data import Dataset

def setup_model():
    model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier without Softmax
    num_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(num_features),
        nn.Linear(num_features, 20)  # No Softmax during training
    )
    
    # Unfreeze classifier layers
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=1000, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()  # Changed from BCELoss
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_dir = 'checkpoints_softmax'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f'device: {device}')
        
        # Training phase
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)  # No need for one-hot encoding
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'accuracy': accuracy
            }, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'accuracy': accuracy
            }, f'{checkpoint_dir}/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break
    
    return model

class VOCDataset(Dataset):
    def __init__(self, root, year, image_set, transform=None):
        self.dataset = datasets.VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=True
        )
        self.transform = transform
        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, annotation = self.dataset[idx]
        
        # Error handling for empty annotations
        if not annotation['annotation']['object']:
            label = 0  # Default to first class if no objects
        else:
            first_obj = annotation['annotation']['object'][0]
            label = self.classes.index(first_obj['name'])
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

if __name__ == '__main__':
    

    # Update data loading section
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = VOCDataset(
        root='./data',
        year='2012',
        image_set='train',
        transform=transform
    )

    val_dataset = VOCDataset(
        root='./data',
        year='2012',
        image_set='val',
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Setup and train model
    model = setup_model()
    train_model(model, train_loader, val_loader)