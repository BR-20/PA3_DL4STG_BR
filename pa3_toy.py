from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

TRENDS = (-1, 0, 1)


def all_neighbourhoods(max_size=8):
    """Every multiset of trends with 1 to max_size members, as (slowing, steady, speeding) counts."""
    return [(a, b, c) for a in range(max_size + 1) for b in range(max_size + 1) for c in range(max_size + 1)
            if 1 <= a + b + c <= max_size]


def q1a_check(n_trials=100, seed=0):
    rng = np.random.default_rng(seed)
    hood_a, hood_b = [-1, 1], [0, 0]
    for _ in range(n_trials):
        w, c = rng.normal(size=2)
        msg_a = sum(w * x + c for x in hood_a)
        msg_b = sum(w * x + c for x in hood_b)
        assert np.isclose(msg_a, msg_b)
    return {"hood_a": hood_a, "hood_b": hood_b, "n_v": 2, "S_v": 0,
            "steady_a": hood_a.count(0), "steady_b": hood_b.count(0), "trials": n_trials}


def q1b_message(K):
    base = K + 1
    return {-1: 1, 0: base, 1: base ** 2}


def q1b_decode(total, K):
    base = K + 1
    slowing = total % base
    steady = (total // base) % base
    speeding = total // base ** 2
    return slowing, steady, speeding


def q1b_check(K=8):
    phi = q1b_message(K)
    triples = all_neighbourhoods(K)
    sums = [a * phi[-1] + b * phi[0] + c * phi[1] for a, b, c in triples]
    for (a, b, c), s in zip(triples, sums):
        assert q1b_decode(s, K) == (a, b, c)
    assert len(set(sums)) == len(triples)
    base_k = {-1: 1, 0: K, 1: K ** 2}
    collision = (K * base_k[-1], 1 * base_k[0])
    assert collision[0] == collision[1]
    nonsteady_ok = all(a * 1 + b * 0 + c * 1 == a + c for a, b, c in all_neighbourhoods(50))
    assert nonsteady_ok
    return {"K": K, "phi": {str(k): v for k, v in phi.items()}, "n_triples": len(triples),
            "n_distinct_sums": len(set(sums)), "max_sum": max(sums),
            "base_K_collision_sum": collision[0], "nonsteady_checked_up_to": 50}


def q1c_check(max_size=8):
    tasks = {"(i) Count steady neighbours": lambda a, b, c: b,
             "(ii) More than half slowing": lambda a, b, c: 2 * a > a + b + c,
             "(iii) Any neighbour slowing": lambda a, b, c: a >= 1}
    methods = {"Sum": lambda a, b, c: (a, b, c),
               "Mean": lambda a, b, c: tuple(Fraction(v, a + b + c) for v in (a, b, c)),
               "Max": lambda a, b, c: tuple(int(v > 0) for v in (a, b, c))}
    yes_rules = {("(i) Count steady neighbours", "Sum"): lambda v: v[1],
                 ("(ii) More than half slowing", "Sum"): lambda v: v[0] > v[1] + v[2],
                 ("(ii) More than half slowing", "Mean"): lambda v: v[0] > Fraction(1, 2),
                 ("(iii) Any neighbour slowing", "Sum"): lambda v: v[0] >= 1,
                 ("(iii) Any neighbour slowing", "Mean"): lambda v: v[0] > 0,
                 ("(iii) Any neighbour slowing", "Max"): lambda v: v[0] == 1}
    hoods = all_neighbourhoods(max_size)
    for (task, method), rule in yes_rules.items():
        assert all(rule(methods[method](*h)) == tasks[task](*h) for h in hoods)
    rows = []
    for task, answer in tasks.items():
        for method, aggregate in methods.items():
            seen = {}
            witness = None
            for h in hoods:
                vec = aggregate(*h)
                if vec in seen and answer(*seen[vec]) != answer(*h):
                    witness = (seen[vec], h, vec)
                    break
                seen.setdefault(vec, h)
            rows.append({"task": task, "method": method, "possible": witness is None,
                         "witness": None if witness is None else
                         {"hood_1": witness[0], "hood_2": witness[1],
                          "vector": [str(v) for v in witness[2]],
                          "answer_1": answer(*witness[0]), "answer_2": answer(*witness[1])}})
    assert {(r["task"], r["method"]) for r in rows if r["possible"]} == set(yes_rules)
    return {"n_neighbourhoods": len(hoods), "rows": rows, "yes_rules_checked": len(yes_rules)}


def ring_features(x):
    return np.stack([x, np.roll(x, 1) + np.roll(x, -1)], axis=1)


def q2a_fit(n=40, pos=20):
    x = np.zeros(n)
    x[pos] = 1.0
    target = np.roll(x, -1)
    coef, *_ = np.linalg.lstsq(ring_features(x), target, rcond=None)
    a, b = coef
    pred = ring_features(x) @ coef
    sse = float(np.sum((pred - target) ** 2))
    assert abs(a - 0.0) < 1e-9 and abs(b - 0.5) < 1e-9 and abs(sse - 0.5) < 1e-9
    return {"a": float(a), "b": float(b), "sse": sse,
            "pred_19_20_21": [float(pred[19]), float(pred[20]), float(pred[21])],
            "true_19_20_21": [float(target[19]), float(target[20]), float(target[21])]}


def half_max_width(row):
    return int(np.sum(row >= row.max() / 2))


def q2a_rollout(a, b, n=40, steps=20, fig_path=None):
    x0 = np.exp(-((np.arange(n) - 20) ** 2) / 8)
    pred_seq, true_seq = [x0], [x0]
    for _ in range(steps):
        pred_seq.append(ring_features(pred_seq[-1]) @ np.array([a, b]))
        true_seq.append(np.roll(true_seq[-1], -1))
    pred_seq, true_seq = np.array(pred_seq), np.array(true_seq)
    assert pred_seq.shape == true_seq.shape == (steps + 1, n)
    for t in range(steps + 1):
        assert int(np.argmax(true_seq[t])) == (20 - t) % n
    if fig_path:
        vmax = max(pred_seq.max(), true_seq.max())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, seq, title in zip(axes, [pred_seq, true_seq], ["Fitted layer from (a)i", "True motion rule"]):
            im = ax.imshow(seq, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("Sensor index i")
        axes[0].set_ylabel("Time step t")
        fig.colorbar(im, ax=axes, label="Slowdown x_i(t)")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return {"pred_peak_sensor_t20": int(np.argmax(pred_seq[-1])), "pred_peak_height_t20": float(pred_seq[-1].max()),
            "true_peak_sensor_t20": int(np.argmax(true_seq[-1])), "true_peak_height_t20": float(true_seq[-1].max()),
            "width_t0": half_max_width(x0), "pred_width_t20": half_max_width(pred_seq[-1]),
            "true_width_t20": half_max_width(true_seq[-1]), "pred_sum_t0": float(x0.sum()),
            "pred_sum_t20": float(pred_seq[-1].sum())}


def q2b_check(n=40, trials=100, seed=0):
    torch.manual_seed(seed)
    kernel = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
    max_err = 0.0
    for _ in range(trials):
        x = torch.randn(n, dtype=torch.float64)
        padded = F.pad(x.view(1, 1, n), (1, 1), mode="circular")
        conv_out = F.conv1d(padded, kernel).view(n)
        max_err = max(max_err, float((conv_out - torch.roll(x, -1)).abs().max()))
        layer_out = 0.0 * x + 0.0 * torch.roll(x, 1) + 1.0 * torch.roll(x, -1)
        max_err = max(max_err, float((layer_out - torch.roll(x, -1)).abs().max()))
    assert max_err < 1e-12
    return {"a": 0.0, "w_minus": 0.0, "w_plus": 1.0, "kernel": [0.0, 0.0, 1.0], "max_abs_error": max_err, "trials": trials}


def history_predict(x_now, x_prev, a=0.0, b=1.0, c=-1.0):
    return a * x_now + b * (np.roll(x_now, 1) + np.roll(x_now, -1)) + c * x_prev


def q2c_check(n=40, trials=100, seed=0):
    rng = np.random.default_rng(seed)
    err_back, err_fwd = 0.0, 0.0
    for _ in range(trials):
        x_prev = rng.normal(size=n)
        x_now = np.roll(x_prev, -1)
        err_back = max(err_back, float(np.abs(history_predict(x_now, x_prev) - np.roll(x_now, -1)).max()))
        x_now = np.roll(x_prev, 1)
        err_fwd = max(err_fwd, float(np.abs(history_predict(x_now, x_prev) - np.roll(x_now, 1)).max()))
    assert err_back < 1e-12 and err_fwd < 1e-12
    return {"a": 0.0, "b": 1.0, "c": -1.0, "max_abs_error_backward": err_back,
            "max_abs_error_forward": err_fwd, "trials": trials}
