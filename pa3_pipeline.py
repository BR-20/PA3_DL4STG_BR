import json
import os
import platform
import warnings

import numpy as np
import pandas as pd
import torch

import pa3_analysis as an
import pa3_data as dt
from pa3_models import STGNN, count_parameters, graph_layer_check
from pa3_train import evaluate, set_seed, train_model

GRAPHS = ["none", "fixed", "adaptive", "hybrid"]
WEEK_TRAIN = "2020-01-06"
WEEK_TEST = "2021-04-05"


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def hardware_string(device):
    cpu = platform.processor() or platform.machine()
    if os.path.exists("/proc/cpuinfo"):
        names = [line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")]
        cpu = names[0] if names else cpu
    gpu = torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "none"
    return {"device": str(device), "gpu": gpu, "cpu": cpu, "cpu_threads": os.cpu_count(),
            "torch": torch.__version__, "python": platform.python_version()}


def q3a(data_dir, fig_dir):
    sd_meta = dt.load_sd_meta(f"{data_dir}/ca_meta.csv")
    raw, rows_added = dt.load_sd_flow_5min(data_dir, sd_meta)
    integer_share = float(np.mean(np.mod(raw.to_numpy()[~np.isnan(raw.to_numpy())], 1) == 0))
    flow = dt.resample_sum_15min(raw)
    out = {"n_sensors": int(flow.shape[1]), "first_stamp": str(flow.index[0]), "last_stamp": str(flow.index[-1]),
           "T_15min": int(len(flow)), "T_5min": int(len(raw)), "raw_integer_share": integer_share,
           "resample_checks": dt.check_resample(raw, flow), "missing": dt.missing_summary(raw, flow, rows_added)}
    assert out["n_sensors"] == dt.N_NODES and out["T_15min"] == 70176
    assert out["first_stamp"] == "2020-01-01 00:00:00" and out["last_stamp"] == "2021-12-31 23:45:00"
    train_missing = flow.loc[:"2020-12-31 23:45"].isna().sum()
    week_cols = list(train_missing.sort_values(kind="stable").index[:3])
    out["week_plot"] = an.plot_week(flow, week_cols, WEEK_TRAIN, f"{fig_dir}/q3a_week.png")
    out["week_sensors"] = week_cols
    return sd_meta, flow, out


def q3b(data_dir, sd_meta, flow, fig_dir):
    adj = dt.load_sd_adjacency(f"{data_dir}/ca_rn_adj.npy", sd_meta)
    order = dt.ordering_check(flow.columns, sd_meta)
    stats, off, out_deg = dt.graph_stats(adj)
    an.plot_adjacency(off, f"{fig_dir}/q3b_adjacency.png")
    an.plot_degree(out_deg, f"{fig_dir}/q3b_degree.png")
    return off, out_deg, {"ordering": order, "stats": stats}


def q3c(off, out_deg, sd_meta):
    node = int(np.argsort(out_deg, kind="stable")[len(out_deg) // 2])
    one_hop, two_hop = dt.hop_neighbourhoods(off, node)
    ids = sd_meta.ID.to_numpy()
    return {"node_index": node, "node_id": str(ids[node]), "one_hop_count": len(one_hop),
            "two_hop_count": len(two_hop), "one_hop_ids": [str(ids[j]) for j in one_hop],
            "two_hop_only_ids": [str(ids[j]) for j in two_hop if j not in one_hop]}


def q4a(flow):
    parts = dt.split_flow(flow)
    stats = dt.training_stats(parts["train"])
    assert stats["train_last_stamp"] == "2020-12-31 23:45:00"
    series = {name: dt.SplitSeries(part, stats["mean"], stats["std"]) for name, part in parts.items()}
    counts = {name: len(s) for name, s in series.items()}
    assert counts == {name: len(parts[name]) - (dt.N_IN - 1) - dt.N_OUT for name in parts}
    checks = {name: dt.window_check(s) for name, s in series.items()}
    target = series["test"].target.double()
    restored = (target - stats["mean"]) / stats["std"] * stats["std"] + stats["mean"]
    norm_roundtrip = float((restored - target).nan_to_num(0).abs().max())
    assert norm_roundtrip < 1e-4
    return series, {"norm": stats, "window_counts": counts, "split_rows": {k: len(v) for k, v in parts.items()},
                    "window_checks": checks, "norm_roundtrip_max_diff": norm_roundtrip}


def model_checks(off, n_nodes, device, cfg, series):
    fixed_norm = dt.row_normalise(off)
    set_seed(0)
    x, _ = series["train"].batch(series["train"].prediction_times[:4], device)
    out = {}
    for graph in GRAPHS:
        model = STGNN(n_nodes, graph, fixed_norm, d=cfg["d"], d_e=cfg["d_e"], alpha=cfg["alpha"]).to(device)
        z = model.encoder(x)
        y = model(x)
        assert z.shape == (4, n_nodes, cfg["d"]) and y.shape == (4, dt.N_OUT, n_nodes)
        entry = {"z_shape": list(z.shape), "y_shape": list(y.shape), "n_params": count_parameters(model)}
        if graph != "none":
            a_hat = model.adjacency().detach()
            assert (a_hat >= 0).all() and torch.allclose(a_hat.sum(1), torch.ones(n_nodes, device=device), atol=1e-5)
            entry["graph_loop_max_diff"] = graph_layer_check(model, x)
            assert entry["graph_loop_max_diff"] < 1e-4
        out[graph] = entry
    return fixed_norm, out


def run_experiments(series, fixed_norm, n_nodes, seeds, device, cfg, ckpt_dir):
    runs, models = [], {}
    for graph in GRAPHS:
        for seed in seeds:
            path = f"{ckpt_dir}/{graph}_seed{seed}.pt"
            if os.path.exists(path):
                saved = torch.load(path, map_location=device)
                assert saved["cfg"] == cfg, f"{path} was trained with a different config"
                model = STGNN(n_nodes, graph, fixed_norm, d=cfg["d"], d_e=cfg["d_e"], alpha=cfg["alpha"]).to(device)
                model.load_state_dict(saved["state"])
                info = saved["info"]
            else:
                model, info = train_model(graph, series["train"], series["val"], fixed_norm, n_nodes, seed, device, cfg)
                torch.save({"state": model.state_dict(), "info": info, "cfg": cfg}, path)
            info["val"] = evaluate(model, series["val"], device)
            info["test"] = evaluate(model, series["test"], device)
            info["cfg"] = dict(cfg)
            runs.append(info)
            if seed == seeds[0]:
                models[graph] = model
            print(graph, seed, {k: round(info[s][k], 2) for s in ("val", "test") for k in ("mae", "rmse")},
                  "time_s", round(info["train_seconds"], 1), "epochs", info["epochs_run"])
    return runs, models


def protocol_table(runs, series):
    """One row per model with every shared setting, asserted identical across the four models."""
    rows = []
    for graph in GRAPHS:
        sel = [r for r in runs if r["graph"] == graph]
        cfg = {k: v for k, v in sel[0]["cfg"].items() if k != "top_k_analysis"}
        assert all(r["cfg"] == sel[0]["cfg"] for r in sel)
        rows.append({"graph": graph, "seeds": tuple(r["seed"] for r in sel), **cfg,
                     **{f"{k}_windows": len(s) for k, s in series.items()}})
    table = pd.DataFrame(rows).set_index("graph")
    assert (table.nunique() == 1).all(), "models differ in protocol"
    return table


def graph_diagnostics(flow, fixed_norm, seeds, ckpt_dir, device, cfg):
    """Self weight, effective neighbours and level mixing gap of every graph used for training."""
    train = flow.loc[:"2020-12-31 23:45"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        level = np.nanmean(train, axis=0)
    has_level = ~np.isnan(level)
    level_filled = np.where(has_level, level, np.nanmean(level))
    graphs = {"fixed": [fixed_norm]}
    for graph in ("adaptive", "hybrid"):
        graphs[graph] = []
        for seed in seeds:
            model = STGNN(fixed_norm.shape[0], graph, fixed_norm, d=cfg["d"], d_e=cfg["d_e"], alpha=cfg["alpha"]).to(device)
            model.load_state_dict(torch.load(f"{ckpt_dir}/{graph}_seed{seed}.pt", map_location=device)["state"])
            graphs[graph].append(model.adjacency().detach().cpu().numpy())
    out = {}
    for graph, mats in graphs.items():
        gaps = [np.abs(m @ level_filled - level_filled)[has_level] for m in mats]
        out[graph] = {"self_weight_mean": float(np.mean([np.diag(m).mean() for m in mats])),
                      "effective_neighbours": float(np.mean([an.effective_neighbours(m) for m in mats])),
                      "level_gap_mean": float(np.mean([g.mean() for g in gaps])),
                      "level_gap_pct_of_mean_level": float(100 * np.mean([g.mean() for g in gaps]) / level[has_level].mean())}
    out["mean_sensor_level"] = float(level[has_level].mean())
    out["sensors_without_training_level"] = int((~has_level).sum())
    return out


def summary_table(runs):
    rows = []
    for graph in GRAPHS:
        sel = [r for r in runs if r["graph"] == graph]
        row = {"Model": an.LABELS[graph][0], "Adjacency": an.LABELS[graph][1], "graph": graph, "n_seeds": len(sel)}
        for split in ("val", "test"):
            for metric in ("mae", "rmse"):
                vals = np.array([r[split][metric] for r in sel])
                row[f"{split}_{metric}_mean"] = float(vals.mean())
                row[f"{split}_{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        secs = np.array([r["train_seconds"] for r in sel])
        row["train_s_mean"] = float(secs.mean())
        row["train_s_std"] = float(secs.std(ddof=1)) if len(secs) > 1 else 0.0
        row["epochs_mean"] = float(np.mean([r["epochs_run"] for r in sel]))
        row["capped_runs"] = int(sum(r["epochs_run"] == r["cfg"]["max_epochs"] for r in sel))
        row["best_epoch_mean"] = float(np.mean([r["best_epoch"] for r in sel]))
        row["s_per_epoch_mean"] = float(np.mean([r["train_seconds"] / r["epochs_run"] for r in sel]))
        row["final_train_loss_mean"] = float(np.mean([r["history"][-1]["train_loss"] for r in sel]))
        rows.append(row)
    return pd.DataFrame(rows)


def q5_q6_graph_analysis(models, runs_seeds, fixed_norm, off, sd_meta, flow, fig_dir, ckpt_dir, device, cfg):
    a_learned = models["adaptive"].adaptive_adjacency().detach().cpu().numpy()
    an.plot_learned(fixed_norm, a_learned, f"{fig_dir}/q5b_learned_adjacency.png")
    train_values = flow.loc[:"2020-12-31 23:45"].to_numpy()
    lat, lng, ids = sd_meta.Lat.to_numpy(), sd_meta.Lng.to_numpy(), sd_meta.ID.to_numpy()
    per_seed = []
    for seed in runs_seeds:
        model = STGNN(off.shape[0], "adaptive", d=cfg["d"], d_e=cfg["d_e"]).to(device)
        model.load_state_dict(torch.load(f"{ckpt_dir}/adaptive_seed{seed}.pt", map_location=device)["state"])
        a_seed = model.adaptive_adjacency().detach().cpu().numpy()
        per_seed.append(an.learned_graph_analysis(a_seed, off, lat, lng, train_values, ids, k=cfg["top_k_analysis"]))
    return per_seed, an.effective_neighbours(fixed_norm)
