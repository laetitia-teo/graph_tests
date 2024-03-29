# small test script to try and reproduce the demos in DeepMind's GraphNet
# library with PyTorch Geometric MetaLayer, which reproduces the behavior
# of a Graph Net.

import collections
import itertools
import time
import os.path as op

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import spatial
import torch
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

from torch_geometric.data import Data
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import from_networkx

from graph_models import EncodeProcessDecode

GPU = False

if GPU:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


DISTANCE_WEIGHT_NAME = "distance"  # The name for the distance edge attribute.

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))


def to_one_hot(indices, max_value, axis=-1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot


def get_node_dict(graph, attr):
    """Return a `dict` of node:attribute pairs from a graph."""
    return {k: v[attr] for k, v in graph.node.items()}


def generate_graph(rand,
                   num_nodes_min_max,
                   dimensions=2,
                   theta=1000.0,
                   rate=1.0):
    """Creates a connected graph.

    The graphs are geographic threshold graphs, but with added edges via a
    minimum spanning tree algorithm, to ensure all nodes are connected.

    Args:
        rand: A random seed for the graph generator. Default= None.
        num_nodes_min_max: A sequence [lower, upper) number of nodes per graph.
        dimensions: (optional) An `int` number of dimensions for the positions.
            Default= 2.
        theta: (optional) A `float` threshold parameters for the geographic
            threshold graph's threshold. Large values (1000+) make mostly trees. Try
            20-60 for good non-trees. Default=1000.0.
        rate: (optional) A rate parameter for the node weight exponential sampling
            distribution. Default= 1.0.

    Returns:
        The graph.
    """
    # Sample num_nodes.
    num_nodes = rand.randint(*num_nodes_min_max)

    # Create geographic threshold graph.
    pos_array = rand.uniform(size=(num_nodes, dimensions))
    pos = dict(enumerate(pos_array))
    weight = dict(enumerate(rand.exponential(rate, size=num_nodes)))
    geo_graph = nx.geographical_threshold_graph(
        num_nodes, theta, pos=pos, weight=weight)

    # Create minimum spanning tree across geo_graph's nodes.
    distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))
    i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
    weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
    mst_graph = nx.Graph()
    mst_graph.add_weighted_edges_from(weighted_edges, weight=DISTANCE_WEIGHT_NAME)
    mst_graph = nx.minimum_spanning_tree(mst_graph, weight=DISTANCE_WEIGHT_NAME)
    # Put geo_graph's node attributes into the mst_graph.
    for i in mst_graph.nodes():
        mst_graph.node[i].update(geo_graph.node[i])

    # Compose the graphs.
    combined_graph = nx.compose_all((mst_graph, geo_graph.copy()))
    # Put all distance weights into edge attributes.
    for i, j in combined_graph.edges():
        combined_graph.get_edge_data(i, j).setdefault(DISTANCE_WEIGHT_NAME,
                                                      distances[i, j])
    return combined_graph, mst_graph, geo_graph


def add_shortest_path(rand, graph, min_length=1):
    """Samples a shortest path from A to B and adds attributes to indicate it.

    Args:
        rand: A random seed for the graph generator. Default= None.
        graph: A `nx.Graph`.
        min_length: (optional) An `int` minimum number of edges in the shortest
            path. Default= 1.

    Returns:
        The `nx.DiGraph` with the shortest path added.

    Raises:
        ValueError: All shortest paths are below the minimum length
    """
    # Map from node pairs to the length of their shortest path.
    pair_to_length_dict = {}
    try:
        # This is for compatibility with older networkx.
        lengths = nx.all_pairs_shortest_path_length(graph).items()
    except AttributeError:
        # This is for compatibility with newer networkx.
        lengths = list(nx.all_pairs_shortest_path_length(graph))
    for x, yy in lengths:
        for y, l in yy.items():
            if l >= min_length:
                pair_to_length_dict[x, y] = l
    if max(pair_to_length_dict.values()) < min_length:
        raise ValueError("All shortest paths are below the minimum length")
    # The node pairs which exceed the minimum length.
    node_pairs = list(pair_to_length_dict)

    # Computes probabilities per pair, to enforce uniform sampling of each
    # shortest path lengths.
    # The counts of pairs per length.
    counts = collections.Counter(pair_to_length_dict.values())
    prob_per_length = 1.0 / len(counts)
    probabilities = [
        prob_per_length / counts[pair_to_length_dict[x]] for x in node_pairs
    ]

    # Choose the start and end points.
    i = rand.choice(len(node_pairs), p=probabilities)
    start, end = node_pairs[i]
    path = nx.shortest_path(
        graph, source=start, target=end, weight=DISTANCE_WEIGHT_NAME)

    # Creates a directed graph, to store the directed path from start to end.
    digraph = graph.to_directed()

    # Add the "start", "end", and "solution" attributes to the nodes and edges.
    digraph.add_node(start, start=True)
    digraph.add_node(end, end=True)
    digraph.add_nodes_from(set_diff(digraph.nodes(), [start]), start=False)
    digraph.add_nodes_from(set_diff(digraph.nodes(), [end]), end=False)
    digraph.add_nodes_from(set_diff(digraph.nodes(), path), solution=False)
    digraph.add_nodes_from(path, solution=True)
    path_edges = list(pairwise(path))
    digraph.add_edges_from(set_diff(digraph.edges(), path_edges), solution=False)
    digraph.add_edges_from(path_edges, solution=True)

    return digraph


def graph_to_input_target(graph):
    """Returns 2 graphs with input and target feature vectors for training.

    Args:
        graph: An `nx.DiGraph` instance.

    Returns:
        The input `nx.DiGraph` instance.
        The target `nx.DiGraph` instance.

    Raises:
        ValueError: unknown node type
    """

    def create_feature(attr, fields):
        return np.hstack([np.array(attr[field], dtype=float) for field in fields])

    input_node_fields = ("pos", "weight", "start", "end")
    input_edge_fields = ("distance",)
    target_node_fields = ("solution",)
    target_edge_fields = ("solution",)

    input_graph = graph.copy()
    target_graph = graph.copy()

    solution_length = 0
    for node_index, node_feature in graph.nodes(data=True):
        input_graph.add_node(
            node_index, features=create_feature(node_feature, input_node_fields))
        target_node = to_one_hot(
            create_feature(node_feature, target_node_fields).astype(int), 2)[0]
        target_graph.add_node(node_index, features=target_node)
        solution_length += int(node_feature["solution"])
    solution_length /= graph.number_of_nodes()

    for receiver, sender, features in graph.edges(data=True):
        input_graph.add_edge(
            sender, receiver, features=create_feature(features, input_edge_fields))
        target_edge = to_one_hot(
            create_feature(features, target_edge_fields).astype(int), 2)[0]
        target_graph.add_edge(sender, receiver, features=target_edge)

    input_graph.graph["features"] = np.array([0.0])
    target_graph.graph["features"] = np.array([solution_length], dtype=float)

    return input_graph, target_graph


def generate_networkx_graphs(rand, num_examples, num_nodes_min_max, theta):
    """Generate graphs for training.

    Args:
        rand: A random seed (np.RandomState instance).
        num_examples: Total number of graphs to generate.
        num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
            graph. The number of nodes for a graph is uniformly sampled within this
            range.
        theta: (optional) A `float` threshold parameters for the geographic
            threshold graph's threshold. Default= the number of nodes.

    Returns:
        input_graphs: The list of input graphs.
        target_graphs: The list of output graphs.
        graphs: The list of generated graphs.
    """
    input_graphs = []
    target_graphs = []
    graphs = []
    for _ in range(num_examples):
        graph = generate_graph(rand, num_nodes_min_max, theta=theta)[0]
        graph = add_shortest_path(rand, graph)
        input_graph, target_graph = graph_to_input_target(graph)
        input_graphs.append(input_graph)
        target_graphs.append(target_graph)
        graphs.append(graph)
    return input_graphs, target_graphs, graphs


def create_placeholders(rand, batch_size, num_nodes_min_max, theta):
    """Creates placeholders for the model training and evaluation.

    Args:
        rand: A random seed (np.RandomState instance).
        batch_size: Total number of graphs per batch.
        num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
            graph. The number of nodes for a graph is uniformly sampled within this
            range.
        theta: A `float` threshold parameters for the geographic threshold graph's
            threshold. Default= the number of nodes.

    Returns:
        input_ph: The input graph's placeholders, as a graph namedtuple.
        target_ph: The target graph's placeholders, as a graph namedtuple.
    """
    # Create some example data for inspecting the vector sizes.
    input_graphs, target_graphs, _ = generate_networkx_graphs(
        rand, batch_size, num_nodes_min_max, theta)
    input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
    target_ph = utils_tf.placeholders_from_networkxs(target_graphs)
    return input_ph, target_ph


def create_feed_dict(rand, batch_size, num_nodes_min_max, theta, input_ph,
                                         target_ph):
    """Creates placeholders for the model training and evaluation.

    Args:
        rand: A random seed (np.RandomState instance).
        batch_size: Total number of graphs per batch.
        num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
            graph. The number of nodes for a graph is uniformly sampled within this
            range.
        theta: A `float` threshold parameters for the geographic threshold graph's
            threshold. Default= the number of nodes.
        input_ph: The input graph's placeholders, as a graph namedtuple.
        target_ph: The target graph's placeholders, as a graph namedtuple.

    Returns:
        feed_dict: The feed `dict` of input and target placeholders and data.
        raw_graphs: The `dict` of raw networkx graphs.
    """
    inputs, targets, raw_graphs = generate_networkx_graphs(
        rand, batch_size, num_nodes_min_max, theta)
    input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)
    target_graphs = utils_np.networkxs_to_graphs_tuple(targets)
    feed_dict = {input_ph: input_graphs, target_ph: target_graphs}
    return feed_dict, raw_graphs


def compute_accuracy(target, output, use_nodes=True, use_edges=False):
    """Calculate model accuracy.

    Returns the number of correctly predicted shortest path nodes and the number
    of completely solved graphs (100% correct predictions).

    Args:
        target: A `graphs.GraphsTuple` that contains the target graph.
        output: A `graphs.GraphsTuple` that contains the output graph.
        use_nodes: A `bool` indicator of whether to compute node accuracy or not.
        use_edges: A `bool` indicator of whether to compute edge accuracy or not.

    Returns:
        correct: A `float` fraction of correctly labeled nodes/edges.
        solved: A `float` fraction of graphs that are completely correctly labeled.

    Raises:
        ValueError: Nodes or edges (or both) must be used
    """
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = []
        if use_nodes:
            c.append(xn == yn)
        if use_edges:
            c.append(xe == ye)
        c = np.concatenate(c, axis=0)
        s = np.all(c)
        cs.append(c)
        ss.append(s)
    correct = np.mean(np.concatenate(cs, axis=0))
    solved = np.mean(np.stack(ss))
    return correct, solved


def create_loss_ops(target_op, output_ops):
    loss_ops = [
        tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes) +
        tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
        for output_op in output_ops
    ]
    return loss_ops


def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]


class GraphPlotter(object):

    def __init__(self, ax, graph, pos):
        self._ax = ax
        self._graph = graph
        self._pos = pos
        self._base_draw_kwargs = dict(G=self._graph, pos=self._pos, ax=self._ax)
        self._solution_length = None
        self._nodes = None
        self._edges = None
        self._start_nodes = None
        self._end_nodes = None
        self._solution_nodes = None
        self._intermediate_solution_nodes = None
        self._solution_edges = None
        self._non_solution_nodes = None
        self._non_solution_edges = None
        self._ax.set_axis_off()

    @property
    def solution_length(self):
        if self._solution_length is None:
            self._solution_length = len(self._solution_edges)
        return self._solution_length

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = self._graph.nodes()
        return self._nodes

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._graph.edges()
        return self._edges

    @property
    def start_nodes(self):
        if self._start_nodes is None:
            self._start_nodes = [
                n for n in self.nodes if self._graph.node[n].get("start", False)
            ]
        return self._start_nodes

    @property
    def end_nodes(self):
        if self._end_nodes is None:
            self._end_nodes = [
                n for n in self.nodes if self._graph.node[n].get("end", False)
            ]
        return self._end_nodes

    @property
    def solution_nodes(self):
        if self._solution_nodes is None:
            self._solution_nodes = [
                n for n in self.nodes if self._graph.node[n].get("solution", False)
            ]
        return self._solution_nodes

    @property
    def intermediate_solution_nodes(self):
        if self._intermediate_solution_nodes is None:
            self._intermediate_solution_nodes = [
                n for n in self.nodes
                if self._graph.node[n].get("solution", False) and
                not self._graph.node[n].get("start", False) and
                not self._graph.node[n].get("end", False)
            ]
        return self._intermediate_solution_nodes

    @property
    def solution_edges(self):
        if self._solution_edges is None:
            self._solution_edges = [
                    e for e in self.edges
                    if self._graph.get_edge_data(e[0], e[1]).get("solution", False)
            ]
        return self._solution_edges

    @property
    def non_solution_nodes(self):
        if self._non_solution_nodes is None:
            self._non_solution_nodes = [
                    n for n in self.nodes
                    if not self._graph.node[n].get("solution", False)
            ]
        return self._non_solution_nodes

    @property
    def non_solution_edges(self):
        if self._non_solution_edges is None:
            self._non_solution_edges = [
                e for e in self.edges
                if not self._graph.get_edge_data(e[0], e[1]).get("solution", False)
            ]
        return self._non_solution_edges

    def _make_draw_kwargs(self, **kwargs):
        kwargs.update(self._base_draw_kwargs)
        return kwargs

    def _draw(self, draw_function, zorder=None, **kwargs):
        draw_kwargs = self._make_draw_kwargs(**kwargs)
        collection = draw_function(**draw_kwargs)
        if collection is not None and zorder is not None:
            try:
                # This is for compatibility with older matplotlib.
                collection.set_zorder(zorder)
            except AttributeError:
                # This is for compatibility with newer matplotlib.
                collection[0].set_zorder(zorder)
        return collection

    def draw_nodes(self, **kwargs):
        """Useful kwargs: nodelist, node_size, node_color, linewidths."""
        if ("node_color" in kwargs and
            isinstance(kwargs["node_color"], collections.Sequence) and
            len(kwargs["node_color"]) in {3, 4} and
            not isinstance(kwargs["node_color"][0],
                           (collections.Sequence, np.ndarray))):
            num_nodes = len(kwargs.get("nodelist", self.nodes))
            kwargs["node_color"] = np.tile(
                np.array(kwargs["node_color"])[None], [num_nodes, 1])
        return self._draw(nx.draw_networkx_nodes, **kwargs)

    def draw_edges(self, **kwargs):
        """Useful kwargs: edgelist, width."""
        return self._draw(nx.draw_networkx_edges, **kwargs)

    def draw_graph(self,
                   node_size=200,
                   node_color=(0.4, 0.8, 0.4),
                   node_linewidth=1.0,
                   edge_width=1.0):
        # Plot nodes.
        self.draw_nodes(
            nodelist=self.nodes,
            node_size=node_size,
            node_color=node_color,
            linewidths=node_linewidth,
            zorder=20)
        # Plot edges.
        self.draw_edges(edgelist=self.edges, width=edge_width, zorder=10)

    def draw_graph_with_solution(self,
                                 node_size=200,
                                 node_color=(0.4, 0.8, 0.4),
                                 node_linewidth=1.0,
                                 edge_width=1.0,
                                 start_color="w",
                                 end_color="k",
                                 solution_node_linewidth=3.0,
                                 solution_edge_width=3.0):
        node_border_color = (0.0, 0.0, 0.0, 1.0)
        node_collections = {}
        # Plot start nodes.
        node_collections["start nodes"] = self.draw_nodes(
            nodelist=self.start_nodes,
            node_size=node_size,
            node_color=start_color,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=100)
        # Plot end nodes.
        node_collections["end nodes"] = self.draw_nodes(
            nodelist=self.end_nodes,
            node_size=node_size,
            node_color=end_color,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=90)
        # Plot intermediate solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.intermediate_solution_nodes]
        else:
            c = node_color
        node_collections["intermediate solution nodes"] = self.draw_nodes(
            nodelist=self.intermediate_solution_nodes,
            node_size=node_size,
            node_color=c,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=80)
        # Plot solution edges.
        node_collections["solution edges"] = self.draw_edges(
            edgelist=self.solution_edges, width=solution_edge_width, zorder=70)
        # Plot non-solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.non_solution_nodes]
        else:
            c = node_color
        node_collections["non-solution nodes"] = self.draw_nodes(
            nodelist=self.non_solution_nodes,
            node_size=node_size,
            node_color=c,
            linewidths=node_linewidth,
            edgecolors=node_border_color,
            zorder=20)
        # Plot non-solution edges.
        node_collections["non-solution edges"] = self.draw_edges(
            edgelist=self.non_solution_edges, width=edge_width, zorder=10)
        # Set title as solution length.
        self._ax.set_title("Solution length: {}".format(self.solution_length))
        return node_collections

###############################################################################

###############################################################################

seed = 1  #@param{type: 'integer'}
rand = np.random.RandomState(seed=seed)

num_examples = 15  #@param{type: 'integer'}
# Large values (1000+) make trees. Try 20-60 for good non-trees.
theta = 20  #@param{type: 'integer'}
num_nodes_min_max = (16, 17)

input_graphs, target_graphs, graphs = generate_networkx_graphs(
    rand, num_examples, num_nodes_min_max, theta)

# num = min(num_examples, 16)
# w = 3
# h = int(np.ceil(num / w))
# fig = plt.figure(40, figsize=(w * 4, h * 4))
# fig.clf()
# for j, graph in enumerate(graphs):
#     ax = fig.add_subplot(h, w, j + 1)
#     pos = get_node_dict(graph, "pos")
#     plotter = GraphPlotter(ax, graph, pos)
#     coll = plotter.draw_graph_with_solution()

# plt.show()

num_examples = 32

###############################################################################

###############################################################################

def from_nx(g):
    """
    Converts a NetworkX graph into the Data objects used by Pytorch Geometric.

    Arguments : 

        - NetworkX graph g

    Returns :

        - data : torch_geometric.data.Data structure representing the metric 
            graph, with node attributes positions, and whether it is a start 
            or a finish node, and edge attributes being the distance between
            two nodes;
        - sol : torch_geometric.data.Data structure representing the solution
            graph, with graph/edge attributes being whether the node/edge 
            belongs to the shortest path between the start and finish nodes.
    """
    X = []
    sol_X = []
    for node in g._node.values():
        x = []
        x.append(node['weight'])
        x.append(node['pos'][0])
        x.append(node['pos'][1])
        x.append(int(node['start']))
        x.append(int(node['end']))
        X.append(x)

        sol_X.append([node['solution']])
    edge_index = []
    edge_attr = []
    sol_edge = []
    for e in g.edges:
        edge_index.append(list(e))
        d = g.get_edge_data(*e)
        edge_attr.append([d['distance']])

        sol_edge.append([int(d['solution'])])
    X = torch.tensor(X)
    sol_X = torch.tensor(sol_X)
    edge_index = torch.tensor(edge_index).T
    edge_attr = torch.tensor(edge_attr)
    sol_edge = torch.tensor(sol_edge)
    y = torch.tensor([[0.]])
    data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)
    sol = Data(x=sol_X, edge_index=edge_index, edge_attr=sol_edge, y=y)

    return data, sol

def compute_accuracy(pred, true_class):
    """
    Computes accuracy of predicted data with the true classes. The arguments
    are expected to be detached from the computation graph.

    Arguments :
        - pred : a float tensor of size [N, 2] with scores for the first class
            ('not on shortest path') and the second class ('on shortest path')
        - true_class : a long tensor of size [N] with the true classes for each
            node.
    """
    pred_class = (pred.T[1] >= pred.T[0]).long()
    accurate = np.logical_not(np.logical_xor(pred_class, true_class))
    return sum(accurate.float()) / len(accurate)

def all_correct(pred, true):
    """
    Returns one if all the elements match, 0 otherwhise. The arguments are
    expected to be detached from the computation graph.

    Arguments :
        - pred : a float tensor of size [N, 2] with scores for the first class
            ('not on shortest path') and the second class ('on shortest path')
        - true_class : a long tensor of size [N] with the true classes for each
            node.
    """
    pred_class = (pred.T[1] >= pred.T[0]).long()
    accurate = np.logical_not(np.logical_xor(pred_class, true_class))
    return accurate.all()

def validation(models, num_nodes_min_max, n_valid, seed=42):
    """
    Validation procedure for a list of models.
    
    Validating consitsts in assessing accuracy for graphs as a function of the
    number of nodes. The validation, for each model and each number of nodes,
    is done on n_valid samples.
    A second procedure is investigating the role of the number of message-
    passing runs on accuracy. 
    Along with mean accuracy, the proportion of completely solved graphs is 
    also recorded.

    Arguments :
        - models : list of models to evaluate
        - num_nodes_min_max : integer tuple, specifying the the range of node
            number for the randomly generated graphs for evaluation.
        - n_valid : the number of graphs, for each node number, on which
            to perform validation.
    """
    acc_stats_allmodels = []
    corr_stats_allmodels = []
    theta = 20
    min_nodes, max_nodes = num_nodes_min_max
    for model in models:
        acc_stats = []
        corr_stats = []
        for nodes in range(min_nodes, max_nodes + 1):
            _, _, graphs = generate_networkx_graphs(
                seed,
                n_valid,
                (nodes, nodes+1),
                theta)
            accuracies = []
            corrects = []
            graphs = list(map(from_nx, graphs))
            for g, sol in graphs:
                x = g.x
                edge_attr = g.edge_attr
                edge_index = g.edge_index
                u = g.y
                batch = torch.zeros(len(x), dtype=torch.long)
                output = model(
                    x, edge_index, edge_attr, u, batch)
                
                true_class = sol.x.long()[:, 0]
                pred = output[-1].detach()
                accuracy = compute_accuracy(pred, true_class)
                correct = float(all_correct(pred, true_class))
                accuracies.append(accuracy)
                corrects.append(correct)
            # mean and other stats
            mean_acc = sum(accuracies) / len(accuracies)
            mean_correct = sum(corrects) / len(corrects)
            acc_stats.append(mean_acc)
            corr_stats.append(mean_correct)
        acc_stats_allmodels.append(acc_stats)
        corr_stats_allmodels.append(corr_stats)
    return acc_stats_allmodels, corr_stats_allmodels


def save_models(models, path, prefix):
    """
    Saves a list of models (their state_dict). All the models are assumed to
    have the same architecture, which is specified in a small text file 
    accompanying the models.

    The function creates a specific directory with the name 'prefix', which is
    also pre-pended to the model name.
    """
    # save model description
    descr = models[0].__repr__()
    with open(op.join(path, prefix, 'descr.txt')) as f:
        f.write(descr)
    # save models
    for i, model in enumerate(models):
        name = prefix + '_model_' + str(i) + '.pt'
        save_path = op.join(path, prefix, name)
        torch.save(model.state_dict(), save_path)

def load_models(path, prefix):
    """
    Loads a list of models saved with the save_models function.
    """
    model = EncodeProcessDecode(f_dict)


def plot_graphs(list_of_lists):
    """
    Plots the graph of the time series generated by the training process.
    """
    for l in list_of_lists:
        plt.plot(l)
    plt.show()

pyg = list(map(from_nx, graphs))

data = [g[0] for g in pyg]
solution = [g[1] for g in pyg]

training_it = 10000

# transform the data into a form we can use
gs = []
true = []

# for g in graphs:
#     x = g.end.float() + d.start.float()
#     edge_index = g.edge_index
#     edge_attr = g.distance
f_e = 1
f_x = 5
f_u = 1
N = 10
h = 5
f_dict = {
    'f_e': f_e,
    'f_x': f_x,
    'f_u': f_u,
    'f_e_out': 2,
    'f_x_out': 2}

# test if model works on forward
# g = data[0]

# x = g.x
# edge_attr = g.edge_attr
# edge_index = g.edge_index
# u = g.y
# batch = torch.tensor(np.zeros(len(x)), dtype=torch.long)
# x_pred, edge_attr_pred, y_pred = model(x, edge_index, edge_attr, u, batch)
# print('x_pred : %s' % x_pred)

# It works !

cross_entropy = CrossEntropyLoss()

S = len(pyg) + 1

def one_pass(e, model, optimizer, graphs, train=True):
    optimizer.zero_grad()
    loss = 0
    acc = 0
    for g, sol in graphs:

        # get data
        x = g.x
        edge_attr = g.edge_attr
        edge_index = g.edge_index
        u = g.y
        batch = torch.tensor(np.zeros(len(x)), dtype=torch.long)

        # predict and compare
        # x_pred is class scores for classes : 'does not lie on shortest path'
        # or 'lies on shortest path'
        # output is a list of predictions, one for each step of preocessing
        output = model(
            x, edge_index, edge_attr, u, batch)
        true_class = sol.x.long()[:, 0]
        loss_x = 0.
        for x_pred, _, _ in output:
            loss_x += cross_entropy(x_pred, true_class)
        # normalize by graph size
        loss_x /= len(x)
        # acc += compute_accuracy(x_pred, sol.x) / S
        # loss_e = cross_entropy(edge_attr_pred, sol.edge_attr.float())
        # loss_x /= S
        # loss_e /= S
        loss += loss_x.item()
        # loss += loss_e.item()
        # loss_x.backward(retain_graph=True)
        if train:
            loss_x.backward()
        acc = 'fix this'
    if train:
        optimizer.step()
    print('epoch : %s, loss : %s, accuracy : %s' % (e, loss, acc))
    return loss, acc

nb_epochs = 5000

###############################################################################

###############################################################################

list_of_losses = []
list_of_accuracies = []
list_of_models = []

random_seeds = np.arange(20)

# different regimes for curriculum learning
training_regimes = [
    (3, 4), (3, 5), (4, 6), (5, 9), (8, 13), (10, 17), (16, 18)]

regime_steps = [100, 200, 300, 500, 1000, 1000, 1000]

def run():
    try:
        for i, seed in enumerate(random_seeds):
            np.random.RandomState(seed=seed)
            torch.manual_seed(seed)

            losses = []
            accuracies = []
            model = EncodeProcessDecode(f_dict)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            # scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
            # name = 'model' + str(i) + '_initial.pt'
            # torch.save(model.state_dict(), op.join('saves', name))

            # for epoch in range(nb_epochs):
            for j in range(len(regime_steps)):
                nb_epochs = regime_steps[j]
                for epoch in range(nb_epochs):
                    _, _, graphs = generate_networkx_graphs(
                        seed,
                        num_examples,
                        training_regimes[j],
                        theta)
                    graphs = list(map(from_nx, graphs))
                    loss, acc = one_pass(epoch, model, optimizer, graphs)
                    losses.append(loss)
                    accuracies.append(acc)

            # name = 'model' + str(i) + '_' + str(nb_epochs) + '.pt'
            # torch.save(model.state_dict(), op.join('saves', name))

            # for g, sol in pyg:
            #     x = g.x
            #     print(g.x)
            #     edge_attr = g.edge_attr
            #     edge_index = g.edge_index
            #     u = g.y
            #     batch = torch.tensor(np.zeros(len(x)), dtype=torch.long)

            #     x, _, _ = model(x, edge_index, edge_attr, u, batch)

            #     print(x)
            #     print(sol.x)

            list_of_losses.append(losses)
            list_of_accuracies.append(accuracies)
            list_of_models.append(model)
    except KeyboardInterrupt:
        list_of_losses.append(losses)
        list_of_models.append(model)

    print('Done ! Hope it worked all right for you.')



# plt.plot(losses)
# plt.show()