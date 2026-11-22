import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pa3_data import N_OUT

LABELS = {"none": ("Temporal-only", "None"), "fixed": ("STGNN", "Fixed"),
          "adaptive": ("STGNN", "Adaptive"), "hybrid": ("STGNN", "Hybrid")}


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(h))


def top_k_edges(a_learned, k):
    scores = a_learned.copy()
    np.fill_diagonal(scores, -np.inf)
    cols = np.argsort(-scores, axis=1)[:, :k]
    rows = np.repeat(np.arange(scores.shape[0]), k)
    return rows, cols.ravel()


def series_correlation(train_values, rows, cols):
    corrs = []
    for i, j in zip(rows, cols):
        both = ~np.isnan(train_values[:, i]) & ~np.isnan(train_values[:, j])
        a, b = train_values[both, i], train_values[both, j]
        if both.sum() > 2 and a.std() > 0 and b.std() > 0:
            corrs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(corrs))


def effective_neighbours(a_hat):
    return float(np.mean(1.0 / np.sum(a_hat ** 2, axis=1)))


def learned_graph_analysis(a_learned, off, lat, lng, train_values, ids, k=10, seed=0):
    """Q6(b): compare the strongest learned links with the road graph, geography and training data."""
    n = off.shape[0]
    rows, cols = top_k_edges(a_learned, k)
    road = off != 0
    in_road = road[rows, cols]
    in_road_either = in_road | road[cols, rows]
    chance = road.sum() / (n * (n - 1))
    chance_either = (road | road.T).sum() / (n * (n - 1))
    dist_learned = haversine_km(lat[rows], lng[rows], lat[cols], lng[cols])
    r_road, c_road = np.nonzero(road)
    dist_road = haversine_km(lat[r_road], lng[r_road], lat[c_road], lng[c_road])
    rng = np.random.default_rng(seed)
    r_rand = rng.integers(0, n, rows.size)
    c_rand = (r_rand + rng.integers(1, n, rows.size)) % n
    dist_rand = haversine_km(lat[r_rand], lng[r_rand], lat[c_rand], lng[c_rand])
    absent = ~in_road_either
    weights = a_learned[rows, cols]
    best_absent = int(np.argmax(np.where(absent, weights, -np.inf))) if absent.any() else None
    example = None
    if best_absent is not None:
        i, j = int(rows[best_absent]), int(cols[best_absent])
        both = ~np.isnan(train_values[:, i]) & ~np.isnan(train_values[:, j])
        example = {"from_id": str(ids[i]), "to_id": str(ids[j]), "weight": float(weights[best_absent]),
                   "distance_km": float(dist_learned[best_absent]),
                   "train_corr": float(np.corrcoef(train_values[both, i], train_values[both, j])[0, 1])}
    return {"k": k, "n_edges": int(rows.size),
            "share_in_road_pct": 100 * float(in_road.mean()), "chance_in_road_pct": 100 * float(chance),
            "share_in_road_either_pct": 100 * float(in_road_either.mean()),
            "chance_in_road_either_pct": 100 * float(chance_either),
            "median_km_learned": float(np.median(dist_learned)), "median_km_road": float(np.median(dist_road)),
            "median_km_random": float(np.median(dist_rand)),
            "absent_from_road_count": int(absent.sum()), "absent_from_road_pct": 100 * float(absent.mean()),
            "corr_learned_absent": series_correlation(train_values, rows[absent], cols[absent]) if absent.any() else None,
            "corr_random": series_correlation(train_values, r_rand, c_rand),
            "max_row_weight_mean": float(np.max(a_learned, axis=1).mean()), "uniform_weight": 1.0 / n,
            "effective_neighbours": effective_neighbours(a_learned),
            "strongest_absent_example": example}


def plot_week(flow, sensor_cols, start, path):
    start_ts = pd.Timestamp(start)
    week = flow.loc[start_ts:start_ts + pd.Timedelta(days=7) - pd.Timedelta(minutes=15)]
    fig, ax = plt.subplots(figsize=(11, 4))
    for col in sensor_cols:
        ax.plot(week.index, week[col], label=f"Sensor {col}", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic flow (vehicles per 15 min)")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    stats = {col: {"min": float(week[col].min()), "max": float(week[col].max()), "mean": float(week[col].mean()),
                   "max_time": str(week[col].idxmax())} for col in sensor_cols}
    return {"start": str(week.index[0]), "end": str(week.index[-1]), "n_steps": len(week), "sensors": stats}


def plot_adjacency(off, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.ma.masked_equal(off, 0.0), cmap="viridis", interpolation="nearest")
    ax.set_facecolor("white")
    ax.set_xlabel("Sensor index j (to)")
    ax.set_ylabel("Sensor index i (from)")
    fig.colorbar(im, ax=ax, label="Road distance weight A_ij")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_degree(out_deg, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(out_deg, bins=np.arange(int(out_deg.max()) + 2) - 0.5, color="#378ADD", edgecolor="white")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("Number of neighbours |N(i)| (out neighbours, self excluded)")
    ax.set_ylabel("Number of sensors")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learned(a_fixed_norm, a_learned, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title in zip(axes, [a_fixed_norm, a_learned], ["Fixed, normalised", "Adaptive, learned"]):
        im = ax.imshow(np.log10(mat + 1e-6), cmap="viridis", interpolation="nearest", vmin=-6, vmax=0)
        ax.set_title(title)
        ax.set_xlabel("Sensor index j")
    axes[0].set_ylabel("Sensor index i")
    fig.colorbar(im, ax=axes, label="log10(weight + 1e-6)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def predict_window(model, series, t_list, device):
    model.eval()
    with torch.no_grad():
        x, _ = series.batch(torch.tensor(t_list), device)
        return (model(x) * series.std + series.mean).cpu().numpy()


def plot_pred_vs_actual(models, series, sensor_idx, sensor_ids, week_start, device, path, step=N_OUT):
    """Prediction made step intervals ahead, plotted at the target time, for one test week."""
    times = series.index
    first = int(times.searchsorted(pd.Timestamp(week_start)))
    targets = np.arange(first, first + 7 * 96)
    origins = targets - step
    assert origins.min() >= int(series.prediction_times[0])
    fig, axes = plt.subplots(len(sensor_idx), 1, figsize=(11, 3 * len(sensor_idx)), sharex=True)
    actual = series.target.numpy()
    preds = {label: predict_window(model, series, origins.tolist(), device)[:, step - 1, :] for label, model in models.items()}
    for ax, n, sid in zip(axes, sensor_idx, sensor_ids):
        ax.plot(times[targets], actual[targets, n], color="black", linewidth=1.2, label="Actual")
        for label, pred in preds.items():
            ax.plot(times[targets], pred[:, n], linewidth=1, label=label)
        ax.set_ylabel("Flow (veh / 15 min)")
        ax.set_title(f"Sensor {sid}")
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Time")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    week_mae = {}
    for label, pred in preds.items():
        err = pred[:, sensor_idx] - actual[targets][:, sensor_idx]
        week_mae[label] = float(np.nanmean(np.abs(err)))
    return {"start": str(times[targets[0]]), "end": str(times[targets[-1]]), "step_ahead": step, "week_mae": week_mae}
