import h5py
import numpy as np
import pandas as pd
import torch

N_NODES = 716
N_IN, N_OUT = 8, 4
SPLITS = {"train": ("2020-01-01", "2021-01-01"),
          "val": ("2021-01-01", "2021-04-01"),
          "test": ("2021-04-01", "2022-01-01")}
STEPS_PER_DAY = 96


def load_sd_meta(meta_path):
    meta = pd.read_csv(meta_path)
    sd_meta = meta[meta.District == 11].reset_index(drop=True)
    assert len(sd_meta) == N_NODES, len(sd_meta)
    assert sd_meta.ID.is_unique and sd_meta.ID2.is_unique
    return sd_meta


def to_datetime_index(raw_index):
    magnitude = abs(int(raw_index[0]))
    unit = "ns" if magnitude > 1e17 else "us" if magnitude > 1e14 else "ms" if magnitude > 1e11 else "s"
    return pd.to_datetime(raw_index, unit=unit)


def read_columns_h5(path, column_ids, rows_per_chunk=4096):
    """Read only the wanted sensor columns from a pandas fixed format HDF5 file."""
    wanted = [str(c).strip() for c in column_ids]
    with h5py.File(path, "r") as f:
        group = f[list(f.keys())[0]]
        if "block0_values" not in group:
            frame = pd.read_hdf(path, columns=wanted)
            return frame[wanted].astype(np.float32)
        n_rows = group["axis1"].shape[0]
        out = np.full((n_rows, len(wanted)), np.nan, dtype=np.float32)
        found = np.zeros(len(wanted), dtype=bool)
        for b in range(int(group.attrs["nblocks"])):
            items = [s.decode().strip() if isinstance(s, bytes) else str(s).strip() for s in group[f"block{b}_items"][...]]
            position = {name: j for j, name in enumerate(items)}
            dst = [k for k, name in enumerate(wanted) if name in position]
            src = [position[wanted[k]] for k in dst]
            if not dst:
                continue
            values = group[f"block{b}_values"]
            for start in range(0, n_rows, rows_per_chunk):
                out[start:start + rows_per_chunk, dst] = values[start:start + rows_per_chunk][:, src]
            found[dst] = True
        assert found.all(), f"{(~found).sum()} sensor columns missing in {path}"
        index = to_datetime_index(group["axis1"][...])
    return pd.DataFrame(out, index=index, columns=wanted)


def load_sd_flow_5min(data_dir, sd_meta, years=(2020, 2021)):
    frames = [read_columns_h5(f"{data_dir}/ca_his_raw_{y}.h5", sd_meta.ID) for y in years]
    raw = pd.concat(frames).sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]
    full_index = pd.date_range("2020-01-01 00:00", "2021-12-31 23:55", freq="5min")
    rows_added = int(len(full_index.difference(raw.index)))
    raw = raw.reindex(full_index)
    assert list(raw.columns) == [str(i) for i in sd_meta.ID]
    return raw, rows_added


def resample_sum_15min(raw):
    """15 minute count = sum of three 5 minute counts, missing unless all three exist."""
    assert len(raw) % 3 == 0 and raw.index[0].minute % 15 == 0
    blocks = raw.to_numpy(dtype=np.float64).reshape(-1, 3, raw.shape[1])
    summed = blocks.sum(axis=1)
    out = pd.DataFrame(summed, index=raw.index[::3], columns=raw.columns)
    assert (out.index[1:] - out.index[:-1] == pd.Timedelta("15min")).all()
    return out


def check_resample(raw, flow, n_checks=3, seed=0):
    rng = np.random.default_rng(seed)
    observed = np.argwhere(~np.isnan(flow.to_numpy()))
    rows = []
    for t, n in observed[rng.choice(len(observed), n_checks, replace=False)]:
        by_hand = float(raw.iloc[3 * t:3 * t + 3, n].sum())
        assert abs(by_hand - flow.iat[t, n]) < 1e-6
        rows.append({"time": str(flow.index[t]), "sensor": flow.columns[n],
                     "five_min_values": raw.iloc[3 * t:3 * t + 3, n].tolist(), "fifteen_min_value": float(flow.iat[t, n])})
    return rows


def missing_summary(raw, flow, rows_added):
    raw_missing = int(raw.isna().to_numpy().sum())
    flow_missing = flow.isna().to_numpy()
    per_sensor = flow_missing.mean(axis=0)
    return {"raw_5min_cells": int(raw.size), "raw_5min_missing": raw_missing,
            "raw_5min_missing_pct": 100 * raw_missing / raw.size, "raw_rows_added_by_reindex": rows_added,
            "flow_15min_cells": int(flow.size), "flow_15min_missing": int(flow_missing.sum()),
            "flow_15min_missing_pct": 100 * flow_missing.mean(),
            "sensors_with_any_missing": int((per_sensor > 0).sum()),
            "sensors_over_50pct_missing": int((per_sensor > 0.5).sum()),
            "max_sensor_missing_pct": 100 * float(per_sensor.max())}


def split_flow(flow):
    parts = {name: flow[(flow.index >= start) & (flow.index < end)] for name, (start, end) in SPLITS.items()}
    assert parts["train"].index[-1] < parts["val"].index[0] and parts["val"].index[-1] < parts["test"].index[0]
    assert sum(len(p) for p in parts.values()) == len(flow)
    return parts


def training_stats(train_part):
    values = train_part.to_numpy()
    observed = values[~np.isnan(values)]
    return {"mean": float(observed.mean()), "std": float(observed.std()),
            "train_last_stamp": str(train_part.index[-1]), "n_values": int(observed.size)}


class SplitSeries:
    """Holds one split: filled normalised inputs and raw targets with missing values kept as NaN."""

    def __init__(self, part, mean, std):
        self.index = part.index
        self.target = torch.tensor(part.to_numpy(), dtype=torch.float32)
        filled = part.ffill().fillna(mean).to_numpy()
        self.inputs = torch.tensor((filled - mean) / std, dtype=torch.float32)
        self.prediction_times = torch.arange(N_IN - 1, len(part) - N_OUT)
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.prediction_times)

    def batch(self, t, device):
        past = t[:, None] + torch.arange(-(N_IN - 1), 1)
        future = t[:, None] + torch.arange(1, N_OUT + 1)
        x = self.inputs[past].to(device)
        y = self.target[future].to(device)
        return x, y


def window_check(series, t=None):
    t = int(series.prediction_times[0]) if t is None else t
    x, y = series.batch(torch.tensor([t]), "cpu")
    assert x.shape == (1, N_IN, series.inputs.shape[1]) and y.shape == (1, N_OUT, series.inputs.shape[1])
    assert torch.equal(x[0], series.inputs[t - N_IN + 1:t + 1])
    assert torch.equal(y[0].nan_to_num(-1), series.target[t + 1:t + N_OUT + 1].nan_to_num(-1))
    first, last = int(series.prediction_times[0]), int(series.prediction_times[-1])
    assert first - (N_IN - 1) == 0 and last + N_OUT == len(series.index) - 1
    return {"t": t, "input_stamps": [str(series.index[t - N_IN + 1]), str(series.index[t])],
            "target_stamps": [str(series.index[t + 1]), str(series.index[t + N_OUT])],
            "x_shape": list(x.shape[1:]) + [1], "y_shape": list(y.shape[1:]) + [1],
            "first_window_start": str(series.index[first - N_IN + 1]), "last_window_end": str(series.index[last + N_OUT])}


def load_sd_adjacency(adj_path, sd_meta):
    ca_adj = np.load(adj_path, mmap_mode="r")
    ids2 = sd_meta.ID2.to_numpy()
    adj = np.asarray(ca_adj[np.ix_(ids2, ids2)], dtype=np.float64)
    assert adj.shape == (N_NODES, N_NODES)
    return adj


def ordering_check(flow_columns, sd_meta):
    data_ids = list(flow_columns)
    adj_ids = [str(i) for i in sd_meta.ID]
    assert data_ids == adj_ids
    return {"first_three": [(data_ids[k], int(sd_meta.ID2[k])) for k in range(3)], "n": len(data_ids)}


def graph_stats(adj):
    diag_nonzero = int(np.count_nonzero(np.diag(adj)))
    off = adj.copy()
    np.fill_diagonal(off, 0.0)
    asym = off != off.T
    out_deg = np.count_nonzero(off, axis=1)
    in_deg = np.count_nonzero(off, axis=0)
    nonzero = off[off != 0]
    return {"nodes": int(adj.shape[0]), "nonzero_edges_offdiag": int(np.count_nonzero(off)),
            "diag_nonzero": diag_nonzero, "symmetric": bool(not asym.any()),
            "max_abs_asym": float(np.abs(off - off.T).max()), "asym_ordered_pairs": int(asym.sum()),
            "asym_unordered_pairs": int(np.triu(asym, 1).sum()),
            "density_pct": 100 * np.count_nonzero(off) / (adj.shape[0] * (adj.shape[0] - 1)),
            "weight_min": float(nonzero.min()), "weight_max": float(nonzero.max()),
            "out_degree_min": int(out_deg.min()), "out_degree_median": float(np.median(out_deg)),
            "out_degree_max": int(out_deg.max()), "out_degree_mean": float(out_deg.mean()),
            "in_degree_min": int(in_deg.min()), "in_degree_max": int(in_deg.max()),
            "isolated_out": int((out_deg == 0).sum())}, off, out_deg


def row_normalise(off):
    """Directed normalisation D_out^-1 (A + I): row i averages over i and its out neighbours."""
    with_self = off + np.eye(off.shape[0])
    normalised = with_self / with_self.sum(axis=1, keepdims=True)
    assert np.allclose(normalised.sum(axis=1), 1.0)
    return normalised


def hop_neighbourhoods(off, node):
    one_hop = set(np.flatnonzero(off[node]).tolist()) - {node}
    two_hop = set(one_hop)
    for j in one_hop:
        two_hop |= set(np.flatnonzero(off[j]).tolist())
    two_hop -= {node}
    assert one_hop <= two_hop
    return sorted(one_hop), sorted(two_hop)
