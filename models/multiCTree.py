import numpy as np
import torch as tc
import higra as hg
import imageio

import math
from torch.nn import Module
from torch.autograd import Function
import matplotlib.pyplot as plt

try:
    from utils import * # imshow, locate_resource
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())

from skimage.filters import threshold_otsu, threshold_li

class ComponentTreeFunction(Function):
  @staticmethod
  def forward(ctx, graph, vertex_weights, tree_type="max", plateau_derivative="full"):
    """
    Construct a component tree of the given vertex weighted graph.

    tree_type must be in ("min", "max", "tos")

    plateau_derivative can be "full" or "single". In the first case, the gradient of an altitude component
    is back-propagated to the vertex weights of the whole plateau (to all proper vertices of the component).
    In the second case, an arbitrary vertex of the plateau is selected and will receive the gradient.

    return: the altitudes of the tree (torch tensor), the tree itself is stored as an attribute of the tensor
    """
    if tree_type == "max":
      tree, altitudes = hg.component_tree_max_tree(graph, vertex_weights.detach().numpy())
    elif tree_type == "min":
      tree, altitudes = hg.component_tree_min_tree(graph, vertex_weights.detach().numpy())
    elif tree_type == "tos":
      tree, altitudes = hg.component_tree_tree_of_shapes_image2d(vertex_weights.detach().numpy())
    else:
      raise ValueError("Unknown tree type " + str(tree_type))

    if plateau_derivative == "full":
      plateau_derivative = True
    elif plateau_derivative == "single":
      plateau_derivative = False
    else:
      raise ValueError("Unknown plateau derivative type " + str(plateau_derivative))
    ctx.saved = (tree, graph, plateau_derivative)
    # altitudes = tc.from_numpy(altitudes).clone().requires_grad_(True)
    altitudes = tc.from_numpy(altitudes.astype(np.float32)).clone().requires_grad_(True)
    # torch function can only return tensors, so we hide the tree as a an attribute of altitudes
    altitudes.tree = tree
    return altitudes

  @staticmethod
  def backward(ctx, grad_output):
    tree, graph, plateau_derivative = ctx.saved
    if plateau_derivative:
      grad_in = grad_output[tree.parents()[:tree.num_leaves()]]
    else:
      leaf_parents = tree.parents()[:tree.num_leaves()]
      _, indices = np.unique(leaf_parents, return_index=True)
      grad_in = tc.zeros((tree.num_leaves(),), dtype=grad_output.dtype)
      grad_in[indices] = grad_output[leaf_parents[indices]]
    return None, hg.delinearize_vertex_weights(grad_in, graph), None

class ComponentTree(Module):
    def __init__(self, tree_type):
        super().__init__()
        tree_types = ("max", "min", "tos")
        if tree_type not in tree_types:
          raise ValueError("Unknown tree type " + str(tree_type) + " possible values are " + " ".join(tree_types))

        self.tree_type = tree_type

    def forward(self, graph, vertex_weights):
        altitudes = ComponentTreeFunction.apply(graph, vertex_weights, self.tree_type)
        return altitudes.tree, altitudes

max_tree = ComponentTree("max")
min_tree = ComponentTree("min")
tos_tree = ComponentTree("tos")

class AccumulateSumFunction(Function):
  """
  Pytorch differentiable function that computes for a tree t and an attribute a (can have requires_grad = True)
  defined on the leaves of t, the new node attribute res defined by:

    res(n) = a(n) if n is a leaf of t
    res(n) = sum_{c in children n} res(c) otherwise
  """
  @staticmethod
  def forward(ctx, tree, leaf_attribute):

    leaf_attribute_np = leaf_attribute.detach().numpy()
    res = hg.accumulate_sequential(tree, leaf_attribute_np, hg.Accumulators.sum)
    res = tc.from_numpy(res).clone().requires_grad_(True)

    ctx.saved = (tree, )

    return res

  @staticmethod
  def backward(ctx, grad_output):
    tree,  = ctx.saved

    grad_output_np = grad_output.detach().clone().numpy()
    res = hg.propagate_sequential_and_accumulate(tree, grad_output_np, hg.Accumulators.sum)
    grad_in = tc.from_numpy(res[:tree.num_leaves()]).clone()
    
  
    return None, grad_in

class AccumulateSum(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, tree, leaf_attribute):
        return AccumulateSumFunction.apply(tree, leaf_attribute)

class AccumulateSumMaxFunction(Function):
  """
  Pytorch differentiable function that computes for a tree t and an attribute a (can have requires_grad = True)
  defined on the nodes of t, the new node attribute res defined by:

    res(n) = a(n) + max_{c in children n} res(c)
  """
  @staticmethod
  def forward(ctx, tree, node_attribute):

    node_attribute_np = node_attribute.detach().numpy()
    res_np = hg.accumulate_and_add_sequential(tree, node_attribute_np, node_attribute_np[:tree.num_leaves()], hg.Accumulators.max)
    res = tc.from_numpy(res_np).clone().requires_grad_(True)

    ctx.saved = (tree, res_np)

    return res

  @staticmethod
  def backward(ctx, grad_output):
    tree, vol_np = ctx.saved

    grad_output_np = grad_output.detach().clone().numpy()

    largest_child = hg.accumulate_parallel(tree, vol_np, hg.Accumulators.argmax)
    child_number = hg.attribute_child_number(tree)
    main_branch = child_number == largest_child[tree.parents()]
    
    res = hg.propagate_sequential_and_accumulate(tree, grad_output_np, hg.Accumulators.sum, main_branch)
    grad_in = tc.from_numpy(res)
    
  
    return None, grad_in

class AccumulateSumMax(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, tree, node_attribute):
        return AccumulateSumMaxFunction.apply(tree, node_attribute)



accum_sum = AccumulateSum()
accum_summax= AccumulateSumMax()






def loss_ranked_selection(saliency_measure, importance_measure, num_positives, p=2, q=2):
  """
  Will try to increase the ranked_measure value of the num_positives first elements to the margin value and decrease the measure on the remaining elements

  :param saliency_measure: 1d torch tensor
  :param importance_measure: torch tensor (same shape as saliency_measure)
  :param num_positive: int >= 0
  :param p: float >= 0 
  :param q: float >= 0 
  :return: a torch scalar
  """
  sorted_indices = tc.argsort(importance_measure, descending=True)
  saliency_measure = saliency_measure[sorted_indices]
  if len(saliency_measure) <= num_positives:
    return -tc.sum(saliency_measure**p)
  else:
    return -tc.sum(saliency_measure[:num_positives]**p) + tc.sum(saliency_measure[num_positives:]**q)
  
  
  
  
  
  
def attribute_depth(tree, altitudes):
  """
  Compute the depth of any node of the tree which is equal to the largest altitude 
  in the subtree rooted in the current node. 

  :param tree: input tree
  :param altitudes: np array (1d), altitudes of the input tree nodes
  :return: np array (1d), depth of the tree nodes
  """
  return hg.accumulate_sequential(tree, altitudes[:tree.num_leaves()], hg.Accumulators.max)

def attribute_saddle_nodes(tree, attribute):
  """
  Let n be a node and let an be an ancestor of n. The node an has a single child node that contains n denoted by ch(an -> n). 
  The saddle and base nodes associated to a node n for the given attribute values are respectively the closest ancestor an  
  of n and the node ch(an -> n) such that there exists a child c of an with attr(ch(an -> n)) < attr(c). 

  :param tree: input tree
  :param attribute: np array (1d), attribute of the input tree nodes
  :return: (np array, np array), saddle and base nodes of the input tree nodes for the given attribute
  """
  max_child_index = hg.accumulate_parallel(tree, attribute, hg.Accumulators.argmax)
  child_index = hg.attribute_child_number(tree)
  main_branch = child_index == max_child_index[tree.parents()]
  main_branch[:tree.num_leaves()] = True

  saddle_nodes = hg.propagate_sequential(tree, np.arange(tree.num_vertices())[tree.parents()], main_branch)
  base_nodes = hg.propagate_sequential(tree, np.arange(tree.num_vertices()), main_branch)
  return saddle_nodes, base_nodes

def attribute_proper_volume(tree, altitudes):
  """
  Compute the proper volume of any node n of the tree t for the node altitudes a defined by:

    vol(n) = area[n]*(a[n]-a[tree.parent(n)]) + max_{c in children n} vol(c) 

  :param tree: input tree
  :param altitudes: np array (1d), altitudes of the input tree nodes
  :return: np array (1d), proper volume of the tree nodes
  """
  parents = tc.from_numpy(tree.parents())
  area = tc.from_numpy(hg.attribute_area(tree))
  
  node_proper_volume = area * (altitudes - altitudes[parents])
  node_proper_volume[:tree.num_leaves()] = 0
  vol = accum_summax(tree, node_proper_volume)
  return vol



def attribute_precision(label, image, tree):
    nbGTPix = label.sum()
    areaNodes = hg.attribute_area(tree)
    gt = np.reshape(label,len(label))
    image = np.reshape(image,len(image))
    att = hg.accumulate_sequential(tree, ((image != 0) & (gt != 0)).astype(int), hg.Accumulators.sum)
    precision = att / areaNodes
    precision[:len(label)] = 0
    return hg.accumulate_and_max_sequential(tree, precision, precision[:len(label)], hg.Accumulators.max)

def otsu_threshold(importance_tensor):
    valid = importance_tensor[(importance_tensor > 0) & (importance_tensor < 1)]
    if valid.numel() == 0:
        return 0.5, valid
    imp_np = valid.detach().cpu().numpy()
    if np.all(imp_np == imp_np[0]):
        return float(imp_np[0]), valid

    return threshold_otsu(imp_np), valid

def li_threshold(importance_tensor):
    importance_tensor = tc.tensor(importance_tensor) if isinstance(importance_tensor, list) else importance_tensor
    valid = importance_tensor[(importance_tensor > 0) & (importance_tensor < 1)]
    if valid.numel() == 0:
        return 0.5
    imp_np = valid.detach().cpu().numpy()
    if np.all(imp_np == imp_np[0]):
        return float(imp_np[0])

    return threshold_li(imp_np)
  
def sigmoid(x, t, lambda_=10):
    x = tc.tensor(x) if isinstance(x, list) else x
    return 1 / (1 + tc.exp(-lambda_ * (x - t)))

def loss_maxima_mCTree(graph, image, label, saliency_measure, importance_measure):
  """
  Loss that favors the presence of num_target_maxima in the given image. 

  
  :param graph: adjacency pixel graph
  :param image: torch tensor 1d, vertex values of the input graph
  :param saliency_measure: string, how the saliency of maxima is measured, can be "altitude" or "dynamics"
  :param importance_measure: string, how the importance of maxima is measured, can be "altitude", "dynamics", "area", or "volume"
  :param num_target_maxima: int >=0, number of maxima that should be present in the result
  :param margin: float >=0, target altitude fo preserved maxima
  :param p: float >=0, power (see parameter p in loss_ranked_selection)
  :param q: float >=0, power (see parameter q in loss_ranked_selection)
  :return: a torch scalar
  """
  if not saliency_measure in ["altitude", "dynamics", "volume"]:
    raise ValueError("Saliency_measure can be either 'altitude' or 'dynamics'")

  if not importance_measure in ["altitude", "dynamics", "area", "volume", "precision"]:
    raise ValueError("Saliency_measure can be either 'altitude', 'dynamics', 'area', 'volume', or 'precision'")
  
  tree, altitudes = max_tree(graph, image)
  altitudes_np = altitudes.detach().numpy()

  extrema = hg.attribute_extrema(tree, altitudes_np)
  extrema_indices = np.arange(tree.num_vertices())[extrema]
  extrema_altitudes = altitudes[tc.from_numpy(extrema_indices)]

  if importance_measure == "dynamics" or saliency_measure == "dynamics":
    depth = attribute_depth(tree, altitudes_np)
    saddle_nodes = tc.from_numpy(attribute_saddle_nodes(tree, depth)[0])
    extrema_dynamics = extrema_altitudes - altitudes[saddle_nodes[extrema_indices]]

  if importance_measure == "area":
    area = hg.attribute_area(tree)
    pass_nodes, base_nodes = attribute_saddle_nodes(tree, area)
    extrema_area = tc.from_numpy(area[base_nodes[extrema_indices]])

  if importance_measure == "volume" or saliency_measure == "volume":
    volume = attribute_proper_volume(tree, altitudes)
    pass_nodes, base_nodes = attribute_saddle_nodes(tree, volume.detach().numpy())
    extrema_volume = volume[tc.from_numpy(base_nodes[extrema_indices])]
    

  if saliency_measure == "altitude":
    saliency = extrema_altitudes
  elif saliency_measure == "dynamics":
    saliency = extrema_dynamics
  elif saliency_measure == "volume":
    saliency = tc.sqrt(extrema_volume)

  if importance_measure == "altitude":
    importance = extrema_altitudes
  elif importance_measure == "dynamics":
    importance = extrema_dynamics
  elif importance_measure == "area":
    importance = extrema_area
  elif importance_measure == "volume":
    importance = extrema_volume
  elif importance_measure == "precision":
        height, width = label.shape  # get from input directly
        label = label.reshape(height * width, 1)
        image = image.detach().numpy().reshape(height * width, 1)
        precision = attribute_precision(label, image, tree)
        extinction_value = hg.attribute_extinction_value(tree, altitudes_np, np.array(precision))
        importance = tc.tensor([extinction_value[i] for i in extrema_indices])
        
  threshold, valid = otsu_threshold(importance)
  
  sigmoid_values = sigmoid(importance, threshold)
  
  return tc.sum((saliency * sigmoid_values))