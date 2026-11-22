import copy
import random
import time

import numpy as np
import torch

from pa3_models import STGNN, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mae(pred, target):
    mask = ~torch.isnan(target)
    if not mask.any():
        return pred.sum() * 0.0
    return (pred[mask] - target[mask]).abs().mean()


def evaluate(model, series, device, batch_size=128):
    """MAE and RMSE in original flow units over every observed target, all 4 steps and all sensors."""
    model.eval()
    abs_sum, sq_sum, count = 0.0, 0.0, 0
    with torch.no_grad():
        for start in range(0, len(series), batch_size):
            t = series.prediction_times[start:start + batch_size]
            x, y = series.batch(t, device)
            pred = model(x) * series.std + series.mean
            mask = ~torch.isnan(y)
            err = pred[mask] - y[mask]
            abs_sum += float(err.abs().sum())
            sq_sum += float((err ** 2).sum())
            count += int(mask.sum())
    return {"mae": abs_sum / count, "rmse": (sq_sum / count) ** 0.5, "n_targets": count}


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def train_model(graph, train_s, val_s, fixed_adj, n_nodes, seed, device, cfg):
    set_seed(seed)
    model = STGNN(n_nodes, graph, fixed_adj, d=cfg["d"], d_e=cfg["d_e"], alpha=cfg["alpha"]).to(device)
    fixed_before = model.a_fixed.clone() if hasattr(model, "a_fixed") else None
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    generator = torch.Generator().manual_seed(seed)
    best_val, best_state, best_epoch, wait, history = float("inf"), None, 0, 0, []
    sync(device)
    start_time = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        order = train_s.prediction_times[torch.randperm(len(train_s), generator=generator)]
        loss_sum, n_batches = 0.0, 0
        for start in range(0, len(order), cfg["batch_size"]):
            x, y = train_s.batch(order[start:start + cfg["batch_size"]], device)
            loss = masked_mae(model(x), (y - train_s.mean) / train_s.std)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            loss_sum += loss.item()
            n_batches += 1
        val = evaluate(model, val_s, device)
        history.append({"epoch": epoch, "train_loss": loss_sum / n_batches, "val_mae": val["mae"]})
        if val["mae"] < best_val:
            best_val, best_state, best_epoch, wait = val["mae"], copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    sync(device)
    train_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    if fixed_before is not None:
        assert torch.equal(fixed_before, model.a_fixed)
    return model, {"graph": graph, "seed": seed, "train_seconds": train_seconds, "epochs_run": len(history),
                   "best_epoch": best_epoch, "n_params": count_parameters(model), "history": history}
