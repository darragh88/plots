"""
graph_dataset_helper.py
-----------------------
Utility for converting a dictionary of `pandas.DataFrame`s—where each key is a
node ID and each DataFrame holds *num_graphs × num_node_features* rows—into a
`torch_geometric.data.InMemoryDataset`.

The resulting dataset can be loaded with `torch_geometric.loader.DataLoader`
for either node‑level or graph‑level supervised learning.

Example
-------
>>> import pandas as pd, numpy as np
>>> from graph_dataset_helper import dict_of_frames_to_dataset
>>>
>>> # Dummy data: 2 graphs, 3 nodes, 2 features
>>> node_frames = {
...     0: pd.DataFrame([[1.0, 0.0], [1.1, 0.1]], columns=['f0', 'f1']),
...     1: pd.DataFrame([[0.5, 2.0], [0.6, 2.1]], columns=['f0', 'f1']),
...     2: pd.DataFrame([[3.0, 3.5], [3.1, 3.6]], columns=['f0', 'f1'])
... }
>>> adj = np.array([[0,1,1],
...                 [1,0,1],
...                 [1,1,0]])
>>> dataset = dict_of_frames_to_dataset(node_frames, adj)
>>> print(dataset)
InMemoryDataset(2)
"""

from __future__ import annotations

from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
from scipy import sparse

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix


def dict_of_frames_to_dataset(
    node_frames: Dict[int, pd.DataFrame],
    adj: Union[np.ndarray, pd.DataFrame],
    *,
    feature_cols: Optional[List[str]] = None,
    label_col: Optional[str] = None,
    graph_level: bool = False,
) -> InMemoryDataset:
    """Convert node‑wise DataFrames to a PyG :class:`InMemoryDataset`.

    Parameters
    ----------
    node_frames
        Mapping ``{node_id: DataFrame}``.  All frames must have **identical**
        shape: ``(num_graphs, num_columns)``.
    adj
        Binary (unweighted) adjacency matrix of shape ``[num_nodes, num_nodes]``.
        The same structure is reused for every graph.
    feature_cols
        Column names to use as node features.  By default *all* columns are
        used, or all except *label_col* if that is given.
    label_col
        Name of the target column.  If *None* no labels are stored.
    graph_level
        When *True*, assume the value of *label_col* is identical across nodes
        and create **graph‑level** labels (shape ``[1]``).  When *False*,
        labels are stored per node (shape ``[num_nodes]``).

    Returns
    -------
    torch_geometric.data.InMemoryDataset
        Each observation (row) becomes one :class:`torch_geometric.data.Data`
        object.

    Notes
    -----
    The column order in *node_frames* must be consistent across nodes.
    """

    # 1 ── deterministic node order
    node_ids = sorted(node_frames.keys())
    num_nodes = len(node_ids)

    # 2 ── column selection
    if feature_cols is None:
        feature_cols = node_frames[node_ids[0]].columns.tolist()
        if label_col is not None and label_col in feature_cols:
            feature_cols.remove(label_col)

    # 3 ── stack features  → (num_graphs, num_nodes, num_features)
    feat_stack = np.stack(
        [node_frames[n][feature_cols].to_numpy(copy=True) for n in node_ids],
        axis=1,
    )
    num_graphs, _, num_features = feat_stack.shape

    # 4 ── labels
    labels = None
    if label_col is not None:
        if graph_level:
            labels = node_frames[node_ids[0]][label_col].to_numpy(copy=True)
        else:
            labels = np.stack(
                [node_frames[n][label_col].to_numpy(copy=True)
                 for n in node_ids],
                axis=1,
            )

    # 5 ── convert adjacency once
    if isinstance(adj, pd.DataFrame):
        adj = adj.values
    edge_index, _ = from_scipy_sparse_matrix(sparse.csr_matrix(adj))

    # 6 ── build Data objects
    data_list = []
    for i in range(num_graphs):
        x_i = torch.tensor(feat_stack[i], dtype=torch.float)  # [nodes, feats]
        data = Data(x=x_i, edge_index=edge_index)

        if labels is not None:
            if graph_level:
                y_i = torch.tensor([labels[i]], dtype=torch.long)
            else:
                y_i = torch.tensor(labels[i], dtype=torch.long)
            data.y = y_i

        data_list.append(data)

    # 7 ── wrap into InMemoryDataset
    class _DictFramesDataset(InMemoryDataset):
        def __init__(self, data_objs):
            super().__init__('.', None)
            self.data, self.slices = self.collate(data_objs)

    return _DictFramesDataset(data_list)


__all__ = ["dict_of_frames_to_dataset"]
