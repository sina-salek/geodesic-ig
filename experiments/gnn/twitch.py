import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import torch_geometric.transforms as T
from torch_geometric.datasets import Twitch
from torch_geometric.nn import GCNConv

from torch_geometric.nn import to_captum_model, to_captum_input
from torch_geometric.explain import Explanation

# from geodesic.utils.pyg_explainer import to_captum, Explainer

import os
import torch

torch.manual_seed(0)
np.random.seed(0)

current_path = os.path.dirname(os.path.realpath(__file__))

dataset = Twitch(root=current_path, name="EN")
data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()

        self.conv1 = GCNConv(dataset.num_features, dim_h)
        self.conv2 = GCNConv(dim_h, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        return F.log_softmax(h, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(64).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    log_logits = model(data.x, data.edge_index)
    loss = F.nll_loss(log_logits, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")


def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1), data.y)
    return acc


acc = test(model, data)
print(f"Test Accuracy: {acc}")

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

# Optional: set number of hops to visualize
# k_hops = 2
# explanation.set_enclosing_subgraph(node_idx, k_hops, data.edge_index)

# Visualize
explanation.visualize_graph(
    # show=True,
    # filepath=None,  # set a path to save the plot
    # node_size=800,
    # node_alpha=0.8,
    # edge_alpha=0.8,
)
