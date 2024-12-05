from torch_geometric.datasets import PPI

import os

from torch_geometric.data import Batch
from torch_geometric.loader import NeighborLoader

from torch_geometric.loader import DataLoader

from torch_geometric.nn import GraphSAGE
import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt


current_path = os.path.dirname(os.path.realpath(__file__))

train_dataset = PPI(root=current_path, split="train")
val_dataset = PPI(root=current_path, split="val")
test_dataset = PPI(root=current_path, split="test")


train_data = Batch.from_data_list(train_dataset)
loader = NeighborLoader(
    train_data,
    batch_size=2048,
    shuffle=True,
    num_neighbors=[20, 10],
    num_workers=2,
    persistent_workers=True,
)

train_loader = DataLoader(train_dataset, batch_size=2)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GraphSAGE(
    in_channels=train_dataset.num_features,
    hidden_channels=512,
    num_layers=2,
    out_channels=train_dataset.num_classes,
).to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def fit():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


from sklearn.metrics import f1_score


@torch.no_grad()
def test(loader):
    model.eval()
    data = next(iter(loader))
    out = model(data.x.to(device), data.edge_index.to(device))
    preds = (out > 0).float().cpu()
    y, pred = data.y.numpy(), preds.numpy()
    return f1_score(y, pred, average="micro") if pred.sum() > 0 else 0


model_path = os.path.join(current_path, "weights")
if not os.path.exists(model_path):
    os.makedirs(model_path)

num_epochs = 301
if len(os.listdir(model_path)) == 0:
    for epoch in range(num_epochs):
        loss = fit()
        val_f1 = test(val_loader)
        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:>3} | Train Loss: {loss:.3f}| Val F1 score: {val_f1:.4f}"
            )
            torch.save(
                model.state_dict(), os.path.join(model_path, f"epoch_{epoch}.pth")
            )
else:
    model.load_state_dict(
        torch.load(os.path.join(model_path, f"epoch_{num_epochs-1}.pth"))
    )


from captum.attr import IntegratedGradients
from torch_geometric.nn import to_captum_model, to_captum_input
from torch_geometric.explain import Explanation

data = next(iter(test_loader)).to(device)

node_idx = 0
mask_type = "node_and_edge"
# captum_model = to_captum(model, mask_type='node_and_edge', output_idx=node_idx)
captum_model = to_captum_model(model, mask_type=mask_type, output_idx=node_idx)
inputs, additional_forward_args = to_captum_input(data.x, data.edge_index, mask_type)
ig = IntegratedGradients(captum_model)

# edge mask is a tensor of shape [num_edges] with values in [0, 1]. In this case, we set it to 1 for all edges.
edge_mask = torch.ones(data.num_edges, requires_grad=True, device=device)
ig_attr = ig.attribute(
    inputs=inputs,
    target=int(data.y[node_idx]),
    additional_forward_args=additional_forward_args,
    internal_batch_size=1,
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
