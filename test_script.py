import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU

from torch_geometric.data import Data
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

# define a small graph

edge_index = torch.tensor([[0, 1, 2],
                           [1, 2, 0]], dtype=torch.long)
x = np.expand_dims(np.array([0., 1., 2.]), -1)
e = np.expand_dims(np.array([0., 0., 0.]), -1)
x = torch.tensor(x, dtype=torch.float)
y = torch.tensor([[0.]])
edge_attr = torch.tensor(e, dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# define target graph

tg_edge_index = torch.tensor([[0, 1, 2],
                              [1, 2, 0]], dtype=torch.long)
tg_x = np.expand_dims(np.array([1., 1., 1.]), -1)
tg_e = np.expand_dims(np.array([1., 1., 1.]), -1)
tg_x = torch.tensor(tg_x, dtype=torch.float)
tg_edge_attr = torch.tensor(e, dtype=torch.float)
tg_y = torch.tensor([1.])

target = Data(x=tg_x, edge_index=tg_edge_index, edge_attr=tg_edge_attr, y=tg_y)

# Define our graph models

f_e = 1
f_x = 1
f_u = 1
h = 4

class EdgeModel(torch.nn.Module):
    def __init__(self,
                 f_e_in,
                 f_x,
                 f_u,
                 h,
                 f_e_out=None):
        """
        Edge model : for each edge, computes the result as a function of the
        edge attribute, the sender and receiver node attribute, and the global
        attribute.

        Arguments :

            - f_e (int): number of edge features
            - f_x (int): number of vertex features
            - f_u (int): number of global features
            - h (int): number of hidden nodes
        """
        super(EdgeModel, self).__init__()
        if f_e_out is None:
            f_e_out = f_e_in
        self.phi_e = Seq(Lin(f_e_in + 2*f_x + f_u, h),
                         ReLU(),
                         Lin(h, f_e_out))

    def forward(self, src, dest, edge_attr, u, batch):
        """
        src [E, f_x] where E is number of edges and f_x is number of vertex
            features : source node tensor
        dest [E, f_x] : destination node tensor
        edge_attr [E, f_e] where f_e is number of edge features : edge tensor
        u [B, f_u] where B is number of batches (graphs) and f_u is number of
            global features : global tensor
        batch [E] : edge-batch mapping
        """
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.phi_e(out)

class NodeModel(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x_in,
                 f_u,
                 h,
                 f_x_out=None):
        """
        Node model : for each node, first computes the mean of every incoming
        edge attibute tensor, then uses this, in addition to the node features
        and the global features to compute the updated node attributes

        Arguments :

            - f_e (int): number of edge features
            - f_x (int): number of vertex features
            - f_u (int): number of global features
            - h (int): number of hidden nodes
        """
        if f_x_out is None:
            f_x_out = f_x_in
        super(NodeModel, self).__init__()
        self.phi_x = Seq(Lin(f_e + f_x_in + f_u, h),
                         ReLU(),
                         Lin(h, f_x_out))

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        """
        row, col = edge_index
        # aggregate all edges which have the same destination
        e_agg_node = scatter_mean(edge_attr, col, dim=0)
        out = torch.cat([x, e_agg_node, u[batch]], 1)
        return self.phi_x(out)

class GlobalModel(torch.nn.Module):
    def  __init__(self,
                 f_e,
                 f_x,
                 f_u_in,
                 h,
                 f_u_out=None):
        """
        Global model : aggregates the edge attributes over the whole graph,
        the node attributes over the whole graph, and uses those to compute
        the next global value.

        Arguments :

            - f_e (int): number of edge features
            - f_x (int): number of vertex features
            - f_u (int): number of global features
            - h (int): number of hidden nodes
        """
        super(GlobalModel, self).__init__()
        if f_u_out is None:
            f_u_out = f_u_in
        self.phi_u = Seq(Lin(f_e + f_x + f_u_in, h),
                         ReLU(),
                         Lin(h, f_u_out))

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        """
        row, col = edge_index
        e_batch = batch[row]
        e_agg = scatter_mean(edge_attr, e_batch, dim=0)
        x_agg = scatter_mean(x, batch, dim=0)
        out = torch.cat([x_agg, e_agg, u], 1)
        return self.phi_u(out)

gn_block = MetaLayer(EdgeModel(f_e, f_x, f_u, h),
                     NodeModel(f_e, f_x, f_u, h),
                     GlobalModel(f_e, f_x, f_u, h))

batch = torch.tensor([0, 0, 0], dtype=torch.long)

# It's alive ! :)
# x, edge_attr, y = gn_block(x, edge_index, edge_attr, y, batch

# Training loop, now that it lives, let's make it suffer a bit

# First define a model that chains N blocks

N = 5

class GraphModel(torch.nn.Module):
    def __init__(self,
                 N,
                 f_e,
                 f_x,
                 f_u,
                 h):
        super(GraphModel, self).__init__()
        self.gn_block = gn_block

    def forward(self, x, edge_index, edge_attr, y, batch):
        for i in range(N):
            x, edge_attr, y = self.gn_block(x, edge_index, edge_attr, y, batch)
        return x, edge_attr, y

# simple target reach

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphModel(N, f_e, f_x, f_u, h)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(200):
    optimizer.zero_grad()
    pr_x, pr_edge_attr, pr_y = model(x, edge_index, edge_attr, y, batch)
    loss_x = F.mse_loss(pr_x, tg_x)
    loss_e = F.mse_loss(pr_edge_attr, tg_edge_attr)
    loss_y = F.mse_loss(pr_y, tg_y)
    loss = loss_x + loss_e + loss_y
    loss.backward()
    optimizer.step()
    print('Epoch : {}; loss : {}'.format(epoch, loss))

print(pr_x, pr_edge_attr, pr_y)
print(tg_x, tg_edge_attr, tg_y)

# Yay ! It's learning !