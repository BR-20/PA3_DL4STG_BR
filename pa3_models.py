import torch
import torch.nn as nn

from pa3_data import N_IN, N_OUT


class TemporalEncoder(nn.Module):
    """Two 1D convolutions over each sensor's 8 past readings, then a linear map to d features."""

    def __init__(self, d):
        super().__init__()
        self.conv1 = nn.Conv1d(1, d, kernel_size=3)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3)
        self.proj = nn.Linear(d * (N_IN - 4), d)

    def forward(self, x):
        batch, steps, nodes = x.shape
        h = x.permute(0, 2, 1).reshape(batch * nodes, 1, steps)
        h = torch.relu(self.conv1(h))
        h = torch.relu(self.conv2(h))
        z = self.proj(h.flatten(1))
        return z.view(batch, nodes, -1)


class STGNN(nn.Module):
    """graph is one of none (temporal only), fixed, adaptive, hybrid."""

    def __init__(self, n_nodes, graph, fixed_adj=None, d=32, d_e=10, alpha=0.5):
        super().__init__()
        self.graph, self.alpha = graph, alpha
        self.encoder = TemporalEncoder(d)
        self.W_g = nn.Linear(d, d, bias=False)
        self.head = nn.Linear(d, N_OUT)
        if graph in ("fixed", "hybrid"):
            self.register_buffer("a_fixed", torch.tensor(fixed_adj, dtype=torch.float32))
        if graph in ("adaptive", "hybrid"):
            self.E1 = nn.Parameter(torch.randn(n_nodes, d_e))
            self.E2 = nn.Parameter(torch.randn(n_nodes, d_e))

    def adaptive_adjacency(self):
        return torch.softmax(torch.relu(self.E1 @ self.E2.T), dim=1)

    def adjacency(self):
        if self.graph == "none":
            return None
        if self.graph == "fixed":
            return self.a_fixed
        if self.graph == "adaptive":
            return self.adaptive_adjacency()
        return self.alpha * self.a_fixed + (1 - self.alpha) * self.adaptive_adjacency()

    def forward(self, x):
        z = self.encoder(x)
        a_hat = self.adjacency()
        mixed = z if a_hat is None else torch.einsum("ij,bjd->bid", a_hat, z)
        h = torch.relu(self.W_g(mixed))
        return self.head(h).permute(0, 2, 1)


def graph_layer_check(model, x):
    """Compare the batched graph layer with a plain loop over batch items, H = ReLU(A Z W_g)."""
    with torch.no_grad():
        z = model.encoder(x)
        a_hat = model.adjacency()
        batched = torch.relu(model.W_g(torch.einsum("ij,bjd->bid", a_hat, z)))
        looped = torch.stack([torch.relu(a_hat @ z[b] @ model.W_g.weight.T) for b in range(x.shape[0])])
    return float((batched - looped).abs().max())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
