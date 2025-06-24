
"""
graph_snapshot_builder.py
-------------------------

Utility functions for turning:

    * a dictionary of per-region pandas DataFrames (node features),
    * a dictionary of per-edge DataFrames (edge features),
    * and a NumPy adjacency matrix that encodes topology,

into a list of `torch_geometric.data.Data` snapshots suitable for
training temporal graph-neural-network models.

The topology is assumed **static** (the same edges apply at every
time-step), while node and edge *values* are allowed to change per
timestamp.
"""

from typing import Dict, List, Tuple, Sequence
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def build_edge_index(adj: np.ndarray, reg_order: Sequence[str]) -> Tuple[torch.Tensor, List[Tuple[str, str]]]:
    src, dst = np.nonzero(adj)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_list = [(reg_order[s], reg_order[d]) for s, d in zip(src, dst)]
    return edge_index, edge_list


def edge_attr_at_ts(edge_dict: Dict[Tuple[str, str], pd.DataFrame],
                    edge_list: Sequence[Tuple[str, str]],
                    ts) -> torch.Tensor:
    rows = []
    for e in edge_list:
        try:
            rows.append(edge_dict[e].loc[ts].to_numpy(dtype=np.float32))
        except KeyError:
            raise KeyError(f"Edge {e} has no data for timestamp {ts}") from None
    return torch.tensor(np.vstack(rows), dtype=torch.float32)


def build_snapshots(region_dfs: Dict[str, pd.DataFrame],
                    edge_dict: Dict[Tuple[str, str], pd.DataFrame],
                    adj: np.ndarray,
                    reg_order: Sequence[str],
                    target: str) -> List[Data]:
    ts_common = sorted(set.intersection(*(set(df.index) for df in region_dfs.values())))
    if not ts_common:
        raise ValueError("No common timestamps across all region DataFrames.")

    edge_index, edge_list = build_edge_index(adj, reg_order)

    snapshots: List[Data] = []
    for ts in ts_common:
        feats, tgts = [], []

        for r in reg_order:
            row = region_dfs[r].loc[ts]
            tgts.append(row[target])
            feats.append(row.drop(target).to_numpy(np.float32))

        x = torch.tensor(np.vstack(feats), dtype=torch.float32)
        y = torch.tensor(tgts, dtype=torch.float32)
        edge_attr = edge_attr_at_ts(edge_dict, edge_list, ts)

        snapshots.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                snap_time=torch.tensor([pd.Timestamp(ts).value])
            )
        )
    return snapshots
