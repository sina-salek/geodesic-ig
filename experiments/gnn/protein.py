import os
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv, GINEConv
from torch_geometric.nn import global_add_pool

seed = 0
torch.manual_seed(seed)

current_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(current_path, "..", "data", "proteins")
dataset = TUDataset(root=data_path, name="PROTEINS")

graph_idx = 0
print(f"Dataset: {dataset}")
print("-----------------------")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of nodes in graph number {graph_idx}: {dataset[graph_idx].x.shape[0]}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

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
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, dataset.num_classes)

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


def train(model, loader, patience=10):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100000
    model.train()

    best_val_loss = float("inf")
    counter = 0

    for epoch in range(epochs + 1):
        total_loss = 0
        acc = 0

        # Train on batches
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss / len(loader)
            acc, f1 = metrics(out.argmax(dim=1), data.y)
            acc = acc / len(loader)
            acc += acc
            loss.backward()
            optimizer.step()

        # Validation
        val_loss, val_acc, val_f1 = test(model, val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:>3} | Train Loss:{total_loss:.2f} | Train Acc: {acc*100:.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f} | Val F1: {val_f1*100:0.2f}%"
            )

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return model


@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc, f1 = metrics(out.argmax(dim=1), data.y)
        acc = acc / len(loader)
        acc += acc
        f1 = f1 / len(loader)
        f1 += f1
    return loss, acc, f1


def metrics(pred_y, y):
    from sklearn.metrics import f1_score, accuracy_score

    accuracy = accuracy_score(y, pred_y)
    f1 = f1_score(y, pred_y, average="macro")
    return accuracy, f1


gin = GIN(dim_h=32)
model_path = os.path.join(current_path, "weights")
if not os.path.exists(model_path):
    os.makedirs(model_path)


if len(os.listdir(model_path)) == 0:
    dataset = dataset.shuffle()
    train_dataset = dataset[: int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8) : int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9) :]


    train_loader = DataLoader(train_dataset, batch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    gin = train(gin, train_loader)
    torch.save(gin.state_dict(), os.path.join(model_path, "gin.pth"))

    torch.save(train_loader, os.path.join(model_path, "train_loader.pth"))
    torch.save(val_loader, os.path.join(model_path, "val_loader.pth"))
    torch.save(test_loader, os.path.join(model_path, "test_loader.pth"))
else:
    gin.load_state_dict(torch.load(os.path.join(model_path, "gin.pth"))) 
    test_loader = torch.load(os.path.join(model_path, "test_loader.pth"))

test_loss, test_acc, test_f1 = test(gin, test_loader)
print(
    f"Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f} | test F1:{test_f1*100:.2f}%"
)


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

PLOTTING = False

if PLOTTING:
   fig, ax = plt.subplots(4, 4)

   for i, data in enumerate(dataset[-16:]):
      out = gin(data.x, data.edge_index, data.batch)
      color = "green" if out.argmax(dim=1) == data.y else "red"
      ix = np.unravel_index(i, ax.shape)
      ax[ix].axis("off")
      G = to_networkx(dataset[i], to_undirected=True)
      nx.draw_networkx(
         G,
         pos=nx.spring_layout(G, seed=seed),
         with_labels=False,
         node_size=10,
         node_color=color,
         width=0.8,
         ax=ax[ix],
      )

   plt.show()


from captum.attr import IntegratedGradients
from torch_geometric.nn import to_captum_model, to_captum_input
from torch_geometric.explain import Explanation
from geodesic.svi_ig import SVI_IG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = dataset[graph_idx].to(device)

# explore the role of node index. Node classification vs graph classification
node_idx = None
mask_type = "node_and_edge"

captum_model = to_captum_model(gin, mask_type=mask_type, output_idx=node_idx)
inputs, additional_forward_args = to_captum_input(data.x, data.edge_index, mask_type)
additional_forward_args = (*additional_forward_args, data.batch)

# ig = IntegratedGradients(captum_model)
ig = SVI_IG(captum_model)
# edge mask is a tensor of shape [num_edges] with values in [0, 1]. In this case, we set it to 1 for all edges.
edge_mask = torch.ones(data.num_edges, requires_grad=True, device=device)


ig_attr = ig.attribute(
    inputs=inputs,
    target=int(data.y[node_idx]),
    additional_forward_args=additional_forward_args,
    internal_batch_size=1,
    n_steps=100,
    # svi_ig parameters
    beta = 1.0,
    num_iterations=1000,
    learning_rate=0.01,
)


# Extract attributions
node_mask = ig_attr[0].squeeze().detach()  # Node attributions
edge_mask = ig_attr[1].squeeze().detach()  # Edge attributions


# Create explanation object
explanation = Explanation(
    node_mask=node_mask,
    edge_mask=edge_mask,
    node_idx=node_idx,  # target node
    edge_index=data.edge_index,
    x=data.x,
    y=data.y[node_idx],
)

explanation.visualize_graph()
