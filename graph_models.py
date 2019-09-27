import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torch.nn import Sequential, Linear, ReLU
from torch.nn import Sigmoid, LayerNorm, Dropout

from torch_geometric.data import Data
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

###############################################################################
#                                                                             #
#                               MLP function                                  #
#                                                                             #
###############################################################################
def mlp_fn(hidden_layer_sizes):
    def mlp(f_in, f_out):
        """
        This function returns a Multi-Layer Perceptron with ReLU non-linearities
        with num_layers layers and h hidden nodes in each layer, with f_in input
        features and f_out output features.
        """
        layers = []
        f1 = f_in
        for f2 in hidden_layer_sizes:
            layers.append(Linear(f1, f2))
            layers.append(ReLU())
            f1 = f2
        layers.append(Linear(f1, f_out))
        # layers.append(ReLU())
        layers.append(LayerNorm(f_out))
        return Sequential(*layers)
    return mlp

###############################################################################
#                                                                             #
#                                 GN Layer                                    #
#                                                                             #
###############################################################################

class EdgeModel(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_e_out=None):
        """
        Edge model : for each edge, computes the result as a function of the
        edge attribute, the sender and receiver node attribute, and the global
        attribute.

        Arguments :

            - f_e (int): number of edge features
            - f_x (int): number of vertex features
            - f_u (int): number of global features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(EdgeModel, self).__init__()
        if f_e_out is None:
            f_e_out = f_e
        self.phi_e = model_fn(f_e + 2*f_x + f_u, f_e_out)

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
                 f_x,
                 f_u,
                 model_fn,
                 f_x_out=None):
        """
        Node model : for each node, first computes the mean of every incoming
        edge attibute tensor, then uses this, in addition to the node features
        and the global features to compute the updated node attributes

        Arguments :

            - f_e (int): number of edge features
            - f_x (int): number of vertex features
            - f_u (int): number of global features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        if f_x_out is None:
            f_x_out = f_x
        super(NodeModel, self).__init__()
        self.phi_x = model_fn(f_e + f_x + f_u, f_x_out)

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
                 f_u,
                 model_fn,
                 f_u_out=None):
        """
        Global model : aggregates the edge attributes over the whole graph,
        the node attributes over the whole graph, and uses those to compute
        the next global value.

        Arguments :

            - f_e (int): number of edge features
            - f_x (int): number of vertex features
            - f_u (int): number of global features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(GlobalModel, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_e + f_x + f_u, f_u_out)

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        """
        row, col = edge_index
        # compute the batch index for all edges
        e_batch = batch[row]
        # aggregate all edges in the graph
        e_agg = scatter_mean(edge_attr, e_batch, dim=0)
        # aggregate all nodes in the graph
        x_agg = scatter_mean(x, batch, dim=0)
        out = torch.cat([x_agg, e_agg, u], 1)
        return self.phi_u(out)

###############################################################################
#                                                                             #
#                              Direct GN Layer                                #
#                                                                             #
###############################################################################

class DirectEdgeModel(torch.nn.Module):
    def __init__(self,
                 f_e,
                 model_fn,
                 f_e_out=None):
        """
        Arguments :
            - f_e (int): number of edge features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(DirectEdgeModel, self).__init__()
        if f_e_out is None:
            f_e_out = f_e
            print(model_fn)
        self.phi_e = model_fn(f_e, f_e_out)

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
        return self.phi_e(edge_attr)

class DirectNodeModel(torch.nn.Module):
    def __init__(self,
                 f_x,
                 model_fn,
                 f_x_out=None):
        """
        Arguments :
            - f_x (int): number of vertex features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(DirectNodeModel, self).__init__()
        if f_x_out is None:
            f_x_out = f_x
        self.phi_x = model_fn(f_x, f_x_out)

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        """
        return self.phi_x(x)

class DirectGlobalModel(torch.nn.Module):
    def  __init__(self,
                 f_u,
                 model_fn,
                 f_u_out=None):
        """
        Arguments :
            - f_u (int): number of global features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(DirectGlobalModel, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_u, f_u_out)

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.phi_u(u)

###############################################################################
#                                                                             #
#                              Reccurent GN                                   #
#                                                                             #
###############################################################################


class EncodeProcessDecode(torch.nn.Module):
    """
    Description taken from DeepMind's graph_nets library

    Full encode-process-decode model.
    The model we explore includes three components:
    - An "Encoder" graph net, which independently encodes the edge, node, and
    global attributes (does not compute relations etc.).
    - A "Core" graph net, which performs N rounds of processing (message-passing)
    steps. The input to the Core is the concatenation of the Encoder's output
    and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
    the processing step).
    - A "Decoder" graph net, which independently decodes the edge, node, and
    global attributes (does not compute relations etc.), on each message-passing
    step.
                        Hidden(t)   Hidden(t+1)
                           |            ^
              *---------*  |  *------*  |  *---------*
              |         |  |  |      |  |  |         |
    Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
              |         |---->|      |     |         |
              *---------*     *------*     *---------*

    Arguments :
    
        - f_dict : feature dictionnary, containing the size of input and 
            output features. The dictionnary should have the following
            fields : f_e (size of edge features), f_x (size of node features),
            f_u (size of global features), f_e_out (optional : size of output
            edge features), f_x_out (optional : size of output node features),
            f_u_out (optional : size of output global features)
            If size of output features is not specified, the input size is
            used.
        - model_fn : a function creating the model to use in each block 
            (e. g. a MLP)
        - N : an int describing the number of message-passing steps to perform
            in the Core. 
    """

    def __init__(self,
                 f_dict,
                 num_layers=2,
                 h=16,
                 N=10):
        super(EncodeProcessDecode, self).__init__()
        f_e, f_x, f_u, f_e_out, f_x_out, f_u_out = self.get_args(f_dict)
        model_fn = mlp_fn([h])
        print(model_fn)
        self.N = N

        self.encoder = MetaLayer(
            DirectEdgeModel(f_e, model_fn, h),
            DirectNodeModel(f_x, model_fn, h),
            DirectGlobalModel(f_u, model_fn))

        # input to the Core is twice as long to account for concatenation
        self.core = MetaLayer(
            EdgeModel(2*h, 2*h, 2*f_u, model_fn, h),
            NodeModel(h, 2*h, 2*f_u, model_fn, h),
            GlobalModel(h, h, 2*f_u, model_fn, f_u))

        self.decoder = MetaLayer(
            DirectEdgeModel(h, model_fn, h),
            DirectNodeModel(h, model_fn, h),
            DirectGlobalModel(f_u, model_fn, f_u))

        model_fn = Linear

        self.transform = MetaLayer(
            DirectEdgeModel(h, model_fn, f_e_out),
            DirectNodeModel(h, model_fn, f_x_out),
            DirectGlobalModel(f_u, model_fn, f_u_out))

    def get_args(self, f_dict):
        f_e, f_x, f_u = f_dict['f_e'], f_dict['f_x'], f_dict['f_u']
        try:
            f_e_out = f_dict['f_e_out']
        except KeyError:
            f_e_out = f_e
        try:
            f_x_out = f_dict['f_x_out']
        except KeyError:
            f_x_out = f_x
        try:
            f_u_out = f_dict['f_u_out']
        except KeyError:
            f_u_out = f_u
        return f_e, f_x, f_u, f_e_out, f_x_out, f_u_out

    def forward(self, x, edge_index, edge_attr, u, batch):
        # first encode the graph with the direct models
        x, edge_attr, u = self.encoder(x, edge_index, edge_attr, u, batch)
        x_h = x
        edge_attr_h = edge_attr
        u_h = u
        
        for i in range(self.N):
            x_cat = torch.cat([x, x_h], 1)
            edge_attr_cat = torch.cat([edge_attr, edge_attr_h], 1)
            u_cat = torch.cat([u, u_h], 1)
            # batch_cat = torch.cat([batch, batch])

            x_h, edge_attr_h, u_h = \
                self.core(x_cat, edge_index, edge_attr_cat, u_cat, batch)

            x_decoded, edge_attr_decoded, y_decoded = self.decoder(
                x_h, edge_index, edge_attr_h, u_h, batch)

        return self.transform(x_h, edge_index, edge_attr_h, u_h, batch)

