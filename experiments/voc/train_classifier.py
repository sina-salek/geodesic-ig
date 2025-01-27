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
    
    # Replace final layer
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, 20)
    
    # Only unfreeze the final layer we just added
    model.classifier[-1].weight.requires_grad = True
    model.classifier[-1].bias.requires_grad = True
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=1000, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
        
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss/len(train_loader):.4f} Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'{checkpoint_dir}/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break

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
            
            # Create one-hot encoded labels
            label = torch.zeros(20)
            for obj in annotation['annotation']['object']:
                class_name = obj['name']
                label[self.classes.index(class_name)] = 1
                
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