import os
from torch_geometric.datasets import TUDataset

current_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(current_path, "..", "data", "proteins")
dataset = TUDataset(root=data_path, name="PROTEINS").shuffle()
print(f"Dataset: {dataset}")
print("-----------------------")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of nodes: {dataset[0].x.shape[0]}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

from torch_geometric.loader import DataLoader

train_dataset = dataset[: int(len(dataset) * 0.8)]
val_dataset = dataset[int(len(dataset) * 0.8) : int(len(dataset) * 0.9)]
test_dataset = dataset[int(len(dataset) * 0.9) :]
print(f"Training set = {len(train_dataset)} graphs")
print(f"Validation set = {len(val_dataset)} graphs")
print(f"Test set = {len(test_dataset)} graphs")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

print("\nTrain loader:")
for i, batch in enumerate(train_loader):
    print(f" - Batch {i}: {batch}")
print("\nValidation loader:")
for i, batch in enumerate(val_loader):
    print(f" - Batch {i}: {batch}")
print("\nTest loader:")
for i, batch in enumerate(test_loader):
    print(f" - Batch {i}: {batch}")

import torch

torch.manual_seed(0)
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool


class GIN(torch.nn.Module):
   def __init__(self, dim_h):
      super(GIN, self).__init__()
      self.conv1 = GINConv(
         Sequential(
               Linear(dataset.num_node_features, dim_h),
               BatchNorm1d(dim_h),
               ReLU(),
               Linear(dim_h, dim_h),
               ReLU(),
         )
      )
      self.conv2 = GINConv(
         Sequential(
               Linear(dim_h, dim_h),
               BatchNorm1d(dim_h),
               ReLU(),
               Linear(dim_h, dim_h),
               ReLU(),
         )
      )
      self.conv3 = GINConv(
         Sequential(
               Linear(dim_h, dim_h),
               BatchNorm1d(dim_h),
               ReLU(),
               Linear(dim_h, dim_h),
               ReLU(),
         )
      )
      self.lin1 = Linear(dim_h*3, dim_h*3)
      self.lin2 = Linear(dim_h*3, dataset.num_classes)

   def forward(self, x, edge_index, batch):
      # Node embeddings
      h1 = self.conv1(x, edge_index)
      h2 = self.conv2(h1, edge_index)
      h3 = self.conv3(h2, edge_index)
      # Graph-level readout
      h1 = global_add_pool(h1, batch)
      h2 = global_add_pool(h2, batch)
      h3 = global_add_pool(h3, batch)
      # Concatenate graph embeddings
      h = torch.cat((h1, h2, h3), dim=1)
      # Classifier
      h = self.lin1(h)
      h = h.relu()
      h = F.dropout(h, p=0.5, training=self.training)
      h = self.lin2(h)
      return F.log_softmax(h, dim=1)
   
def train(model, loader):
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
   epochs = 100
   model.train()
   for epoch in range(epochs+1):
      total_loss = 0
      acc = 0
      val_loss = 0
      val_acc = 0
      # Train on batches
   for data in loader:
      optimizer.zero_grad()
      out = model(data.x, data.edge_index, data.batch)
      loss = criterion(out, data.y)
      total_loss += loss / len(loader)
      acc += accuracy(out.argmax(dim=1), data.y) /len(loader)
      loss.backward()
      optimizer.step()
      # Validation
      val_loss, val_acc = test(model, val_loader)
      # Print metrics every 20 epochs
      if(epoch % 20 == 0):
         print(f'Epoch {epoch:>3} | Train Loss:{total_loss:.2f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
      return model