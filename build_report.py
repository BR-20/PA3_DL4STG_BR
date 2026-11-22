import argparse
import json
import os

import pandas as pd
import pypandoc

R = "results"
FIG = "figures"


def load(name):
    path = f"{R}/{name}"
    if not os.path.exists(path):
        return None
    if name.endswith(".csv"):
        return pd.read_csv(path, keep_default_na=False)
    with open(path) as f:
        return json.load(f)


def f2(x):
    return f"{x:,.2f}"


def pct(x):
    return f"{x:.2f}%"


def hood(counts):
    a, b, c = counts
    items = ["−1"] * a + ["0"] * b + ["+1"] * c
    return "{" + ", ".join(items) + "}"


def join_ids(ids):
    if not ids:
        return "no sensors"
    return ids[0] if len(ids) == 1 else ", ".join(ids[:-1]) + " and " + ids[-1]


def rerun_sentence(previous, table):
    if previous is None:
        return ""
    old = previous.set_index("graph")["test_mae_mean"]
    new = table.set_index("graph")["test_mae_mean"]
    gap = float((old - new).abs().max())
    spread = float(table["test_mae_std"].max())
    return (f"An earlier full run with the same seeds and settings gave mean test MAE values that differ from Table 4 by at most {f2(gap)} vehicles, "
            f"against a largest seed standard deviation of {f2(spread)}, because GPU kernels are not bit for bit deterministic, so a rerun reproduces the numbers within the seed spread rather than exactly. ")


def gen_gap_sentence(rows):
    """Approximate gap between final training MAE and test MAE in flow units, from saved files."""
    gaps = {g: rows[g]["test_mae_mean"] - rows[g]["final_train_loss_mean"] * rows[g]["norm_std"] for g in rows}
    return (f"A rough measure of this is the gap between the training error and the test error, taking the last epoch training loss times the training standard deviation. "
            f"This gap is about {f2(gaps['adaptive'])} vehicles for the adaptive graph model, against {f2(gaps['hybrid'])} for the hybrid graph model, "
            f"{f2(gaps['fixed'])} for the fixed graph model and {f2(gaps['none'])} for the temporal only model, so the learned graph likely fits patterns of 2020 that hold less well in 2021. ")


def geography_verdict(agg):
    ratio = agg["median_km_learned"] / agg["median_km_random"]
    road = agg["median_km_learned"] / agg["median_km_road"]
    if ratio < 0.5:
        return f"so learned links clearly favour nearby sensors, at {ratio:.2f} times the random length. "
    if ratio < 0.95:
        return (f"so learned links are somewhat shorter than random pairs, at {ratio:.2f} times their length, "
                f"but {road:.1f} times longer than road links, so geography pulls the learned graph only weakly. ")
    if ratio <= 1.05:
        return f"so learned links are about as long as random pairs, at {ratio:.2f} times their length, and do not follow geography. "
    return f"so learned links are longer than random pairs, at {ratio:.2f} times their length, and favour distant sensors. "


def road_verdict(agg):
    ratio = agg["share_in_road_pct"] / agg["chance_in_road_pct"]
    if ratio > 1.2:
        return (f"which is {ratio:.1f} times the chance rate, so the learned graph favours road links, "
                f"yet {pct(100 - agg['share_in_road_pct'])} of its strongest links are not road links in that direction. ")
    return f"which is {ratio:.1f} times the chance rate, so the learned graph does not favour road links. "


def corr_verdict(agg, ex):
    diff = agg["corr_learned_absent"] - agg["corr_random"]
    text = (f"The difference in correlation is {diff:.3f}, ")
    if diff > 0.05:
        text += "so these links likely reflect shared traffic patterns such as similar daily commute shapes. "
    elif diff > 0:
        text += "which is small, so shared daily shape explains at most a small part of why these links were learned. "
    else:
        text += "so these links are not explained by shared traffic patterns in this measure. "
    if ex and ex["train_corr"] < agg["corr_random"]:
        text += "The strongest example even has a lower correlation than the random average, so raw correlation alone does not explain the learned weights. "
    return text


def prose_name(label):
    return "temporal only model" if label.startswith("Temporal") else label.split()[-1].lower() + " graph model"


def yes_no(v):
    return ("Yes" if v else "No") if isinstance(v, bool) else str(v)


def mean_std(row, key, n_seeds):
    if n_seeds > 1:
        return f"{f2(row[key + '_mean'])} ± {f2(row[key + '_std'])}"
    return f2(row[key + "_mean"])


def compare(name_a, row_a, name_b, row_b, key, n_seeds):
    """Two sentences comparing the mean of metric key for model a against model b, with a seed noise check."""
    a, b = row_a[key + "_mean"], row_b[key + "_mean"]
    diff = a - b
    word = "lower" if diff < 0 else "higher"
    text = (f"The {name_a} has a test MAE that is {f2(abs(diff))} vehicles {word} than the {name_b}, "
            f"a relative change of {pct(100 * abs(diff) / b)}. ")
    if n_seeds > 1:
        spread = row_a[key + "_std"] + row_b[key + "_std"]
        text += (f"This gap is {'larger than' if abs(diff) > spread else 'within'} the summed seed spread of {f2(spread)} vehicles. ")
    else:
        text += "This gap comes from a single seed, so it is not tested against seed noise. "
    return text


def q_line(text):
    return f"**Q. {text}**\n\n**Answer.**\n\n"


def figure(path, caption):
    return f"![]({path})\n\n{caption}\n\n"


def build(args):
    toy, data, runs = load("toy_results.json"), load("data_results.json"), load("runs.json")
    table, graph, pred = load("summary.csv"), load("graph_analysis.json"), load("pred_plot.json")
    diag = load("diagnostics.json")
    previous = load("previous_summary.csv")
    md = []
    md.append("# AI651 Deep Learning for Space, Time and Graphs, Assignment 3\n\n")
    md.append(f"Name: {args.name or 'PENDING'}\n\nRoll number: {args.roll or 'PENDING'}\n\n")
    md.append(f"Code repository: <{args.repo}>\n\n" if args.repo else "Code repository: PENDING\n\n")

    md.append("# Question 1. What does a sum remember?\n\n")
    q1a, q1b, q1c = toy["q1a"], toy["q1b"], toy["q1c"]
    md.append(q_line("(a) First, use an affine message ϕ(x) = wx + c. As in a GCN, apply nonlinearity after aggregation; omit degree normalisation here. Express the aggregated message using the number of neighbours nv = |N(v)| and their total trend Sv = Σu∈N(v) xu. Explain why equal nv and Sv force identical outputs for every choice of w, c and ρ. Construct two such neighbourhoods with different numbers of steady neighbours."))
    md.append("The affine message is applied to each neighbour and the results are added, so the constant and the slope separate:\n\n"
              "$$\\sum_{u\\in N(v)}\\phi(x_u)=\\sum_{u\\in N(v)}(w x_u+c)=w\\sum_{u\\in N(v)}x_u+c\\,|N(v)|=w S_v+c\\,n_v.$$\n\n"
              "The layer output is therefore $h_v=\\rho(w S_v+c\\,n_v)$, so the only input that ρ ever receives is one number built from $n_v$ and $S_v$. "
              "Two neighbourhoods with the same $n_v$ and the same $S_v$ give the same number $w S_v+c\\,n_v$ for every choice of w and c. "
              "Any function ρ of one equal input returns one equal output. "
              f"The neighbourhoods {{−1, +1}} and {{0, 0}} both have $n_v={q1a['n_v']}$ and $S_v={q1a['S_v']}$, "
              f"yet the first has {q1a['steady_a']} steady neighbours and the second has {q1a['steady_b']}. "
              f"A NumPy check over {q1a['trials']} random pairs (w, c) found equal aggregated messages for both neighbourhoods every time.\n\n")

    md.append(q_line("(b) Now explore how much information a nonlinear message can carry while still being a single number. Choose its three outputs directly; no network training is needed. For this part, assume 1 ≤ |N(v)| ≤ K, where K is a known positive integer. Construct ϕ(−1), ϕ(0) and ϕ(1) in terms of K so that the sum of the neighbours' messages uniquely determines the numbers of slowing, steady and speeding-up neighbours. Explain how to recover all three counts. For K = 8, verify your construction in NumPy on every possible triple of trend counts. To get started, think of a uniquely decodable encoding that stores the three counts in separate place values of one number. How large must the base be to prevent carries? For a simpler target, suppose we only want to count non-steady neighbours (−1 or +1). Choose a message ϕ whose sum gives this count for any neighbourhood size."))
    md.append("Let $a$, $b$ and $c$ be the numbers of slowing, steady and speeding neighbours. The message stores each count in its own place value of base $K+1$:\n\n"
              "$$\\phi(-1)=1,\\qquad \\phi(0)=K+1,\\qquad \\phi(+1)=(K+1)^2,\\qquad \\sum_{u\\in N(v)}\\phi(x_u)=a+b\\,(K+1)+c\\,(K+1)^2.$$\n\n"
              "Each count is at most $|N(v)|\\le K$, so every count fits in one digit of base $K+1$ and no digit can carry into the next place. "
              "A smaller base fails because a count of K would overflow a single digit. "
              f"With base K and K = {q1b['K']}, eight slowing neighbours give $8\\times 1={q1b['base_K_collision_sum']}$, which equals one steady neighbour with message 8, so the two neighbourhoods cannot be told apart. "
              "The counts are read back digit by digit with integer division:\n\n"
              "$$a=s \\bmod (K+1),\\qquad b=\\left\\lfloor \\frac{s}{K+1}\\right\\rfloor \\bmod (K+1),\\qquad c=\\left\\lfloor \\frac{s}{(K+1)^2}\\right\\rfloor.$$\n\n"
              f"For K = {q1b['K']} the messages are {q1b['phi']['-1']}, {q1b['phi']['0']} and {q1b['phi']['1']}. "
              f"The NumPy check enumerated all {q1b['n_triples']} count triples with $1\\le a+b+c\\le 8$, decoded every sum back to its own triple "
              f"and found {q1b['n_distinct_sums']} distinct sums, with the largest sum equal to {q1b['max_sum']}. "
              "For the non steady count the message $\\phi(x)=x^2$ gives 1 for −1 and +1 and 0 for a steady neighbour, "
              "so the sum equals $a+c$ for any neighbourhood size and needs no bound K. "
              f"This identity was also checked in code for every neighbourhood with up to {q1b['nonsteady_checked_up_to']} neighbours.\n\n")

    md.append(q_line("(c) Finally, compare what three aggregation methods let the layer compute. Return to the same road-sensor setting: one to eight neighbours, each reporting −1, 0 or +1. For this part only, replace the scalar message from (b) with these fixed three-coordinate messages: ϕ(−1) = (1, 0, 0), ϕ(0) = (0, 1, 0), ϕ(+1) = (0, 0, 1). This is called one-hot encoding: each coordinate flags one trend (slowing, steady or speeding up). The sensor readings are still scalars; only their messages are now vectors. Use these same messages with each method below. Sums, means and maxima operate separately on each coordinate: Sum: Σu∈N(v) ϕ(xu), Mean: (1/|N(v)|) Σu∈N(v) ϕ(xu), Max: maxu∈N(v) ϕ(xu). In every case, ρ receives only the aggregated vector. It is not separately told the number of neighbours. Complete the table in your report. For each row, decide whether the method can solve the task correctly for every allowed neighbourhood. Make each reason explicit: For Yes, state which coordinate(s) ρ uses and give the calculation or test that produces the answer. For No, give two allowed neighbourhoods with the same aggregated vector but different correct answers. Write that vector and both answers to show why ρ cannot distinguish them. You may reuse the same example pair to justify more than one “No”."))
    yes_reason = {("(i) Count steady neighbours", "Sum"): "ρ reads the steady coordinate, which equals the steady count $s_2$.",
                  ("(ii) More than half slowing", "Sum"): "ρ uses all three coordinates and answers Yes when $s_1>s_2+s_3$.",
                  ("(ii) More than half slowing", "Mean"): "ρ reads the slowing fraction and answers Yes when $m_1>1/2$.",
                  ("(iii) Any neighbour slowing", "Sum"): "ρ answers Yes when $s_1\\ge 1$.",
                  ("(iii) Any neighbour slowing", "Mean"): "ρ answers Yes when $m_1>0$.",
                  ("(iii) Any neighbour slowing", "Max"): "ρ answers Yes when the slowing coordinate $M_1=1$."}
    md.append("In every row ρ receives only the aggregated three coordinate vector and is never told the number of neighbours, "
              "so a method can answer a task only if that vector alone fixes the answer. ")
    md.append("Table 1. This table gives each task and aggregation method, whether ρ can always answer correctly and the reason. "
              "Coordinates are ordered as slowing, steady and speeding. Every verdict was decided by an exhaustive NumPy check over all "
              f"{q1c['n_neighbourhoods']} neighbourhoods with one to eight members. All {q1c['yes_rules_checked']} Yes rules were also checked against the true answer on the same set.\n\n")
    pdf_task = {"(i) Count steady neighbours": "(i) Count steady neighbours",
                "(ii) More than half slowing": "(ii) Decide whether more than half of the neighbours are slowing",
                "(iii) Any neighbour slowing": "(iii) Decide whether any neighbour is slowing"}
    md.append("| Task | Method | Possible? | Reason |\n|---|---|---|---|\n")
    for row in q1c["rows"]:
        if row["possible"]:
            reason = yes_reason[(row["task"], row["method"])]
        else:
            w = row["witness"]
            reason = (f"{hood(w['hood_1'])} and {hood(w['hood_2'])} both give ({', '.join(w['vector'])}), "
                      f"but the correct answers are {yes_no(w['answer_1'])} and {yes_no(w['answer_2'])}.")
        md.append(f"| {pdf_task[row['task']]} | {row['method']} | {'Yes' if row['possible'] else 'No'} | {reason} |\n")
    md.append("\nTable 1 shows that the sum keeps the most information because it keeps the counts themselves. "
              "The mean keeps only proportions, so it loses how many neighbours there are. The max keeps only which trends are present. "
              "The same choice appears in a GCN when it divides by the degree. Dividing makes nodes with many and few neighbours look alike, which helps when only the mix of values matters and hurts when the count itself matters. "
              "The trade off is that the mean and the max are bounded by one, which keeps layer outputs on a similar scale for sensors with very different numbers of neighbours, while the sum grows with the neighbourhood size.\n\n")

    md.append("# Question 2. Which way is the slowdown moving?\n\n")
    fit, roll, q2b, q2c = toy["q2a_fit"], toy["q2a_rollout"], toy["q2b"], toy["q2c"]
    md.append(q_line("(a) A graph layer on the ring that treats both neighbours alike predicts x̂i(t + 1) = a xi(t) + b(xi−1(t) + xi+1(t)). The same two coefficients a, b are used at every sensor. i. Start with one snapshot: sensor 20 reads 1 and all other sensors read 0. Find a, b that minimise the sum of squared prediction errors across all 40 sensors for the next snapshot. Show your calculation. To get started, write the predicted and correct readings at sensors 19, 20 and 21; all other predictions are zero."))
    md.append("Only sensors 19, 20 and 21 have a nonzero prediction or target. The layer predicts $\\hat x_{19}=b$, $\\hat x_{20}=a$ and $\\hat x_{21}=b$, while the true next snapshot is 1 at sensor 19 and 0 at sensors 20 and 21. "
              "The squared error is\n\n$$E(a,b)=(b-1)^2+a^2+b^2.$$\n\n"
              "Setting both partial derivatives to zero gives $\\partial E/\\partial a=2a=0$, so $a=0$. Setting $\\partial E/\\partial b=2(b-1)+2b=0$ gives $b=1/2$. "
              "The minimum error is $E(0,1/2)=1/4+0+1/4=1/2$. "
              f"A least squares fit over all 40 sensors in NumPy returned a = {f2(fit['a'])}, b = {f2(fit['b'])} and a squared error of {f2(fit['sse'])}, which matches the hand result. "
              "The fit splits the slowdown equally between sensors 19 and 21 because a layer that treats both neighbours alike cannot prefer either side.\n\n")
    md.append(q_line("(a) ii. Because the same coefficients are used everywhere, moving the isolated slowdown to another sensor would give the same fitting problem. Keep the fitted coefficients fixed: they now define a layer that can be applied to any snapshot. Explore what this layer does to a wider pattern: start from the smooth bump xi(0) = e^(−(i−20)²/8) and predict 20 steps, using each predicted snapshot as the next input. Separately, shift the same initial bump using the true motion rule. Plot the two sequences as heatmaps: sensor index on the horizontal axis, time t = 0, . . . , 20 on the vertical axis, and the same colour scale for both. In one or two sentences, compare how the bump moves and changes shape. You may use np.roll(x, -1) for one true step."))
    md.append(figure(f"{FIG}/q2a_heatmaps.png", "Figure 1. The plot shows the bump predicted by the fitted layer in the left panel and the bump moved by the true rule in the right panel for 20 steps on one colour scale."))
    md.append(f"Figure 1 shows that the true bump slides one sensor backward per step and reaches sensor {roll['true_peak_sensor_t20']} at t = 20 with its height of {f2(roll['true_peak_height_t20'])} and its width of {roll['true_width_t20']} sensors above half height unchanged, "
              f"while the fitted layer keeps the peak at sensor {roll['pred_peak_sensor_t20']} and flattens it to a height of {f2(roll['pred_peak_height_t20'])} spread over {roll['pred_width_t20']} sensors. "
              f"The predicted bump keeps its total of {f2(roll['pred_sum_t20'])} because the two neighbour weights add to one, so the layer only diffuses the slowdown in both directions instead of moving it.\n\n")
    md.append(q_line("(b) Next, weight the two directions separately: x̂i(t + 1) = a xi(t) + w− xi−1(t) + w+ xi+1(t). Choose a, w−, w+ that give the exact next snapshot for every input, and verify by substitution. Here the layer is told which neighbour is ahead (i + 1) and which is behind (i − 1). Explain why this extra information removes the difficulty in (a). How would you implement this as a 1D convolution? Give the three kernel entries in the order that multiplies [xi−1, xi, xi+1], and state how to handle the ring boundary."))
    md.append("The choice $a=0$, $w_-=0$ and $w_+=1$ gives $\\hat x_i(t+1)=0\\cdot x_i(t)+0\\cdot x_{i-1}(t)+1\\cdot x_{i+1}(t)=x_{i+1}(t)=x_i(t+1)$, which is the true rule for every input and every sensor. "
              "In part (a) one shared weight b multiplied both neighbours, so sensors 19 and 21 always received the same prediction and the fit had to split the slowdown. "
              "Separate weights tell the layer which neighbour is ahead, so it can copy from the sensor ahead and ignore the one behind. "
              "A directed road graph uses the same idea, because each edge records which detector feeds which. "
              "As a 1D convolution the kernel entries in the order that multiplies $[x_{i-1},x_i,x_{i+1}]$ are $[0,0,1]$. "
              "The ring is handled with circular padding of one value on each side, so sensor 39 is placed before sensor 0 and sensor 0 after sensor 39. "
              "PyTorch Conv1d computes a cross correlation and does not flip the kernel, so the entries are used in the order given. "
              f"A check on {q2b['trials']} random snapshots found a largest absolute difference of {q2b['max_abs_error']:.1e} between the circular convolution and np.roll(x, -1).\n\n")
    md.append(q_line("(c) Finally, try using history while keeping the neighbours' weights equal: x̂i(t + 1) = a xi(t) + b(xi−1(t) + xi+1(t)) + c xi(t − 1). Choose new coefficients a, b, c; they need not match part (a). i. The previous and current snapshots come from the stated backward-motion rule. Find a, b, c that predict the next snapshot exactly for every such pair. Show why they work. Hint: where has the reading previously at sensor i moved to at time t? Use this to rewrite xi(t − 1) in terms of a current reading."))
    md.append("Under the backward rule the reading at sensor i at time t − 1 has moved one sensor back by time t, so $x_{i-1}(t)=x_i(t-1)$. "
              "Substituting this into the layer gives\n\n$$\\hat x_i(t+1)=a\\,x_i(t)+b\\,\\big(x_{i-1}(t)+x_{i+1}(t)\\big)+c\\,x_{i-1}(t).$$\n\n"
              "With $a=0$, $b=1$ and $c=-1$ the two $x_{i-1}(t)$ terms cancel and $\\hat x_i(t+1)=x_{i+1}(t)$, which is the true next reading $x_i(t+1)$. "
              f"A check on {q2c['trials']} random pairs gave a largest absolute error of {q2c['max_abs_error_backward']:.1e}, which is floating point rounding.\n\n")
    md.append(q_line("(c) ii. Suppose instead the pattern moves forwards: xi(t + 1) = xi−1(t). The previous snapshot now also follows this forward-motion rule. Check whether the same coefficients from (i) still work, showing your substitution."))
    md.append("Under the forward rule the reading at sensor i at time t − 1 has moved one sensor ahead by time t, so $x_{i+1}(t)=x_i(t-1)$. "
              "With the same coefficients\n\n$$\\hat x_i(t+1)=x_{i-1}(t)+x_{i+1}(t)-x_i(t-1)=x_{i-1}(t)+x_{i+1}(t)-x_{i+1}(t)=x_{i-1}(t),$$\n\n"
              "which is the true forward next reading $x_i(t+1)=x_{i-1}(t)$. The same coefficients therefore still work. "
              f"A check on {q2c['trials']} random forward pairs gave a largest absolute error of {q2c['max_abs_error_forward']:.1e}.\n\n")
    md.append(q_line("(c) iii. Explain how the previous snapshot helps the model distinguish the two directions, even though the spatial neighbours still have equal weights."))
    md.append("Under the backward rule the reading that sat at sensor i one step ago has moved to sensor i − 1, so $x_i(t-1)=x_{i-1}(t)$. "
              "The history term therefore cancels the neighbour behind, which has just received sensor i's old reading. It leaves the neighbour ahead, whose reading moves into sensor i next. "
              "Under the forward rule the same term equals $x_{i+1}(t)$, so it cancels the neighbour ahead and keeps the neighbour behind, which is again the one whose reading arrives next. "
              "The direction is therefore read from the history instead of from the spatial weights. "
              "The single snapshot layer in part (a) had no such record, so the two neighbours stayed indistinguishable however its weights were set. "
              "The trade off is that the history model needs two snapshots and an exact motion rule, while the direction weights in part (b) need only one snapshot but must be told which neighbour is ahead.\n\n")

    if data is None:
        md.append("# Questions 3 to 7\n\nNot executed. The LargeST results files are missing.\n\n")
    else:
        md += real_data_sections(data, runs, table, graph, pred, diag, previous)
    with open("report.md", "w") as f:
        f.write("".join(md))
    pypandoc.convert_file("report.md", "docx", outputfile=args.out,
                          extra_args=["--from=markdown-implicit_figures", f"--resource-path=.:{FIG}"])
    print("wrote", args.out)


def real_data_sections(data, runs, table, graph, pred, diag, previous):
    q3a, q3b, q3c, q4a = data["q3a"], data["q3b"], data["q3c"], data["q4a"]
    cfg, hw, seeds, mc = data["cfg"], data["hardware"], data["seeds"], data["model_checks"]
    miss, st = q3a["missing"], q3b["stats"]
    n_seeds = len(seeds)
    device_text = f"one {hw['gpu']} GPU" if hw["gpu"] != "none" else "the CPU only"
    seeds_text = join_ids([str(x) for x in seeds])
    rows = {r["graph"]: r for r in table.to_dict("records")}
    for r in rows.values():
        r["norm_std"] = q4a["norm"]["std"]
    capped = int(sum(r["capped_runs"] for r in rows.values()))
    capped_runs = [r for r in runs if r["epochs_run"] == cfg["max_epochs"]]
    still_falling = sum(r["history"][-1]["val_mae"] < r["history"][-6]["val_mae"] for r in capped_runs)
    cap_text = (f"{capped} of the {len(runs)} runs stopped at the cap of {cfg['max_epochs']} epochs instead of by early stopping. "
                f"In {still_falling} of these runs the validation MAE was still falling over the last five epochs, so their errors are for this fixed training budget and longer training would likely lower them. " if capped else "")
    md = []
    md.append("# Summary of the traffic experiments\n\n")
    md.append(f"Questions 3 to 7 forecast one hour of traffic flow at {q3a['n_sensors']} San Diego sensors from the previous two hours. "
              f"Four models share one temporal encoder and one forecast head and differ only in how sensors are linked. "
              "On the April to December 2021 test period the mean test MAE in vehicles per fifteen minutes is "
              + ", ".join(f"{f2(rows[g]['test_mae_mean'])} for the {prose_name(rows[g]['Model'] + ' ' + rows[g]['Adjacency'])}"
                          for g in sorted(rows, key=lambda g: rows[g]["test_mae_mean"])[:-1])
              + " and " + (lambda g: f"{f2(rows[g]['test_mae_mean'])} for the {prose_name(rows[g]['Model'] + ' ' + rows[g]['Adjacency'])}")(max(rows, key=lambda g: rows[g]["test_mae_mean"]))
              + ", from best to worst. The hybrid against adaptive order is the least certain of these with three seeds. "
              "The answers to Q4(c), Q6(b) and Q6(c) explain these results with measured numbers.\n\n")
    md.append("# Setup\n\n")
    md.append("Table 2. This table gives the data, split, model and training settings shared by every experiment in Questions 3 to 7.\n\n")
    md.append("| Setting | Value |\n|---|---|\n")
    for k, v in [("Dataset", "LargeST San Diego, District 11, 716 sensors, 2020 and 2021"),
                 ("Sampling", "5 minute counts summed to 15 minute counts"),
                 ("Splits", "Train Jan to Dec 2020, validation Jan to Mar 2021, test Apr to Dec 2021"),
                 ("Window", "8 past steps of 2 hours in, 4 future steps of 1 hour out"),
                 ("Normalisation", f"Training mean {f2(q4a['norm']['mean'])} and std {f2(q4a['norm']['std'])} vehicles per 15 min"),
                 ("Hidden size d", str(cfg["d"])), ("Embedding size d_e", str(cfg["d_e"])),
                 ("Hybrid weight α", str(cfg["alpha"])), ("Loss", "Masked MAE on normalised values"),
                 ("Optimiser", f"Adam, learning rate {cfg['lr']}, batch {cfg['batch_size']}"),
                 ("Epochs", f"At most {cfg['max_epochs']}, early stop after {cfg['patience']} epoch{'s' if cfg['patience'] != 1 else ''} without validation MAE gain"),
                 ("Seeds", ", ".join(str(s) for s in seeds)),
                 ("Hardware", f"GPU {hw['gpu']}, CPU {hw['cpu']} with {hw['cpu_threads']} threads, PyTorch {hw['torch']}")]:
        md.append(f"| {k} | {v} |\n")
    md.append("\n")

    md.append("# Question 3. Dataset Preparation and Graph Construction\n\n")
    md.append(q_line("Q3(a) Data Extraction and Resampling. Download LargeST and extract the San Diego traffic sensor measurements for 2020 and 2021. The original traffic measurements are sampled every five minutes. Convert them into 15-minute intervals. For an average traffic measurement, X15min_t = (1/3) Σ_{k=0}^{2} X5min_{3t+k}. If the measurements represent interval vehicle counts rather than average flow rates, use summation instead. Document your choice and ensure that the aggregation preserves the intended physical units. Required: 1. Extract all 716 San Diego sensors. 2. Retain observations from 2020 and 2021. 3. Resample the traffic measurements to 15-minute intervals. 4. Identify missing values and explain how they are handled. 5. Plot the traffic measurements for three selected sensors over one week."))
    chk = q3a["resample_checks"][0]
    md.append("The source files are ca_meta.csv, ca_rn_adj.npy, ca_his_raw_2020.h5 and ca_his_raw_2021.h5 from the LargeST Kaggle dataset. ")
    md.append(f"The San Diego subset is the set of sensors with District 11 in the LargeST metadata file, which gives {q3a['n_sensors']} sensors, the same rule used by the LargeST repository. "
              f"The raw files hold {q3a['T_5min']:,} five minute steps from {q3a['first_stamp']} onward. The resampled series holds {q3a['T_15min']:,} steps from {q3a['first_stamp']} to {q3a['last_stamp']}. "
              "PeMS flow comes from loop detectors that count the vehicles crossing them, so each five minute value is likely a count. "
              f"In the downloaded files {pct(100 * q3a['raw_integer_share'])} of the observed raw values are whole numbers, which is consistent with counts. "
              "Each fifteen minute value is therefore the sum of three five minute counts, so the unit becomes vehicles per fifteen minutes. "
              f"For example, sensor {chk['sensor']} at {chk['time']} has five minute counts {', '.join(f'{v:g}' for v in chk['five_min_values'])}, which sum to {chk['fifteen_min_value']:g}. "
              "The LargeST repository averages the three readings instead (resample to 15 minutes with the mean in its process_ca_his notebook), so the errors in this report are about three times larger in scale than errors on its averaged data and should not be compared with published LargeST numbers directly.\n\n")
    md.append(f"The raw data has {miss['raw_5min_missing']:,} missing five minute values, which is {pct(miss['raw_5min_missing_pct'])} of all cells. "
              f"A fifteen minute value is marked missing unless all three of its five minute counts exist, because a partial sum would undercount the vehicles. "
              f"This rule gives {miss['flow_15min_missing']:,} missing fifteen minute values, which is {pct(miss['flow_15min_missing_pct'])}, "
              f"with {miss['sensors_over_50pct_missing']} sensors missing more than half of their values and a worst sensor missing {pct(miss['max_sensor_missing_pct'])}. "
              "Model inputs fill a gap with the last observed value from the same split. A gap at the start of a split takes the training mean, so no input uses a future value or another split. "
              + (f"{diag['sensors_without_training_level']} sensors have no observed value in 2020, so their training inputs are the training mean only. They stay in the validation and test scores wherever their targets are observed. " if diag else "")
              + "Missing values are the NaN cells of the raw files. Zero counts are kept as real readings. "
              "Missing targets are excluded from the loss and from every metric, so no filled value is ever scored as a real count. "
              "Another option is to remove a missing sensor from the graph input and adjust the degrees of its neighbours. That option was not tested here.\n\n")
    md.append(figure(f"{FIG}/q3a_week.png", f"Figure 2. The plot shows fifteen minute flow for sensors {join_ids(q3a['week_sensors'])} from {q3a['week_plot']['start']} to {q3a['week_plot']['end']}."))
    wk = q3a["week_plot"]["sensors"]
    md.append("Figure 2 shows the three training sensors with the fewest missing values over the first full Monday to Sunday week of 2020. "
              + " ".join(f"Sensor {sid} ranges from {v['min']:,.0f} to {v['max']:,.0f} vehicles per fifteen minutes with a weekly mean of {f2(v['mean'])} and its highest value at {v['max_time']}."
                         for sid, v in wk.items()) + "\n\n")

    md.append(q_line("Q3(b) Constructing the Traffic Graph. The LargeST dataset provides a road-distance adjacency matrix describing relationships between traffic sensors. Extract the adjacency submatrix corresponding to the 716 San Diego sensors: ASD ∈ R716×716. You must ensure that the sensor ordering in the adjacency matrix matches the ordering of sensors in the traffic measurement tensor. Required: 1. Report the number of nodes and nonzero edges. 2. Determine whether the adjacency matrix is symmetric. 3. Plot the adjacency matrix as a heatmap. 4. Compute the number of neighbors of each sensor and plot its distribution. 5. Explain what a nonzero entry Aij represents."))
    md.append("The adjacency rows and columns are taken with the ID2 index of each San Diego sensor in metadata order. The data columns are taken with the ID of the same sensors in the same order. "
              f"A code check asserts that the {q3b['ordering']['n']} data column IDs and adjacency IDs agree position by position. "
              f"Two independent facts support the alignment. The {st['nonzero_edges_offdiag']:,} off diagonal edges match the edge count listed for the LargeST SD subset in the Torch Spatiotemporal documentation. "
              f"Road linked sensors also lie a median of {f2(graph[0]['median_km_road'])} km apart against {f2(graph[0]['median_km_random'])} km for random pairs. "
              f"The graph has {st['nodes']} nodes and {st['nonzero_edges_offdiag']:,} nonzero edges between different sensors, which is a density of {pct(st['density_pct'])}. "
              f"The raw diagonal has {st['diag_nonzero']} nonzero self entries, which are removed before the edge count and before normalisation. ")
    if st["symmetric"]:
        md.append("The matrix is symmetric, with a largest difference between $A_{ij}$ and $A_{ji}$ of zero.\n\n")
    else:
        md.append(f"The matrix is not symmetric, with {st['asym_unordered_pairs']:,} sensor pairs whose two directions differ and a largest difference of {st['max_abs_asym']:.4f}.\n\n")
    md.append(figure(f"{FIG}/q3b_adjacency.png", "Figure 3. The plot shows the San Diego road distance adjacency with self entries removed, rows as from and columns as to."))
    md.append(figure(f"{FIG}/q3b_degree.png", "Figure 4. The plot shows how many sensors have each number of out neighbours."))
    md.append(f"Figure 3 shows all {st['nonzero_edges_offdiag']:,} nonzero weights, which fill only {pct(st['density_pct'])} of the matrix. "
              f"Figure 4 counts the neighbours of sensor i as $N(i)=\\{{j: A_{{ij}}\\neq 0\\}}$ without i itself. "
              f"The count ranges from {st['out_degree_min']} to {st['out_degree_max']} with a median of {st['out_degree_median']:g} and a mean of {f2(st['out_degree_mean'])}. "
              f"There are {st['isolated_out']} sensors with no out neighbour, which means their rows of A are empty. "
              f"Counted by columns instead, the number of in neighbours ranges from {st['in_degree_min']} to {st['in_degree_max']}. "
              f"$A_{{ij}}$ is the weight of the edge from sensor i to sensor j, so rows are the from side and columns are the to side. "
              f"A nonzero entry $A_{{ij}}$ means that sensor j lies within the LargeST road distance cutoff of sensor i along the road network, with weights between {st['weight_min']:.4f} and {st['weight_max']:.4f}. "
              "The weight likely falls as the road distance grows, as in the Gaussian kernel with a cutoff shown in the assignment, so it measures how strongly traffic at one sensor is expected to relate to traffic at the other. "
              "Road distance is directed because traffic runs one way on many segments, so the distance from i to j can differ from the distance from j to i, which likely explains the asymmetry.\n\n")

    md.append(q_line("Q3(c) Understanding Graph Neighborhoods. Consider a sensor vi. Its one-hop neighborhood is N(i) = {j : Aij ≠ 0}. A two-hop neighborhood includes nodes reachable from i through paths containing at most two edges. Select one sensor from the dataset and identify its one-hop and two-hop neighbors. Answer the following: Why might neighboring sensors help predict traffic at the selected sensor? What is the difference between one-hop and two-hop information aggregation? Could using too many graph convolution layers negatively affect performance? Explain."))
    md.append(f"Sensor {q3c['node_id']} has the median number of neighbours. It has {q3c['one_hop_count']} one hop neighbours, which are {join_ids(q3c['one_hop_ids'])}. "
              f"Its two hop set holds {q3c['two_hop_count']} sensors and adds {join_ids(q3c['two_hop_only_ids'])} to the one hop set. "
              "Neighbouring sensors help because vehicles pass from one sensor to the next along the same road, so a rise in flow upstream arrives a few minutes later. A queue that forms downstream spreads back toward the selected sensor. "
              "A sensor a few kilometres along the same carriageway carries the same stream of vehicles, so it can be more useful than a closer sensor on another road. "
              "One hop aggregation mixes each sensor with its direct neighbours in one graph layer, while two hop aggregation needs two stacked layers or one layer with the squared adjacency. Two hop aggregation also brings in the neighbours of those neighbours. "
              "Distant sensors therefore exchange information through intermediate sensors, one hop per layer, in the same way that pixels reach each other through stacked convolutions. "
              "Too many graph layers can hurt because every layer averages a sensor with its neighbours, so after many layers the features of nearby sensors become almost the same, an effect called oversmoothing (Li, Han and Wu, 2018). "
              "Deep stacks also bring in distant sensors whose traffic has little to do with the selected sensor and add parameters that are harder to train.\n\n")

    md.append("# Question 4. Spatio-Temporal Traffic Forecasting\n\n")
    wc, ws = q4a["window_counts"], q4a["window_checks"]
    md.append(q_line("Q4(a) Preparing Forecasting Windows. For each prediction time t, construct Xt = [xt−7, xt−6, . . . , xt] and the prediction target Yt = [xt+1, xt+2, xt+3, xt+4], where each xt ∈ R716×1. Use chronological splits: Training: January–December 2020, Validation: January–March 2021, Testing: April–December 2021. Normalize using statistics computed only from the training set. Ensure that forecasting windows do not cross split boundaries."))
    md.append(f"Each window takes eight past steps as input with shape (8, 716, 1) and the next four steps as target with shape (4, 716, 1). "
              f"Windows are cut separately inside each split, so a split with L steps gives L − 11 windows: {wc['train']:,} for training, {wc['val']:,} for validation and {wc['test']:,} for testing. "
              f"The first test window reads {ws['test']['input_stamps'][0]} to {ws['test']['input_stamps'][1]} and predicts {ws['test']['target_stamps'][0]} to {ws['test']['target_stamps'][1]}. "
              "Code asserts that the first and last windows of every split start and end inside that split. "
              f"The mean of {f2(q4a['norm']['mean'])} and the standard deviation of {f2(q4a['norm']['std'])} vehicles per fifteen minutes come only from the observed 2020 values, whose last stamp is {q4a['norm']['train_last_stamp']}.\n\n")

    md.append(q_line("Q4(b) Implementing the Forecasting Architecture. Implement a lightweight spatio-temporal forecasting model with three components. Component 1, Temporal encoder: apply a temporal convolution to each sensor's historical observations to obtain a feature representation Z = ftemporal(X), with Z ∈ RB×N×d, where B is the batch size and d is the hidden feature dimension. Component 2, Spatial graph convolution: apply graph convolution to the temporal representations, H = ReLU(ÂZWg), where Â is a normalized adjacency matrix. The graph multiplication operates across the node dimension, independently for each batch item. Component 3, Forecast head: use a linear layer or a small MLP to predict the next four traffic measurements for every sensor, Ŷ = fhead(H). Implementation requirements: 1. Implement a temporal convolutional encoder. 2. Implement at least one graph convolution layer. 3. Implement a forecasting head. 4. Train using MAE or MSE loss. 5. Report validation and test MAE and RMSE in the original traffic-flow units. For CPU feasibility, use hidden dimensions of 16–32 and a lightweight architecture. More complex architectures are not required."))
    md.append("Table 3. This table gives the layers of the forecasting model and their output shapes for a batch of B windows.\n\n")
    md.append("| Component | Layers | Output shape |\n|---|---|---|\n"
              f"| Temporal encoder | Conv1d 1 to {cfg['d']} channels, kernel 3, ReLU, Conv1d {cfg['d']} to {cfg['d']}, kernel 3, ReLU, linear {cfg['d'] * 4} to {cfg['d']} | (B, 716, {cfg['d']}) |\n"
              f"| Graph convolution | $H=\\mathrm{{ReLU}}(\\hat A Z W_g)$, $W_g$ of size {cfg['d']} × {cfg['d']} | (B, 716, {cfg['d']}) |\n"
              f"| Forecast head | Linear {cfg['d']} to 4 | (B, 4, 716) |\n\n")
    f_row = rows["fixed"]
    md.append(f"Table 3 shows that the encoder reads each sensor's own eight steps with the same weights for every sensor, the graph layer mixes sensors with the normalised adjacency $\\hat A$ and the head maps each sensor's features to four predictions. "
              "The temporal encoder is a small temporal convolution network. The two 1D convolutions read short local shapes in each sensor's eight steps. The linear layer then turns them into one feature vector per sensor. "
              "The model is a factorized composition at network level. The temporal operator runs on each sensor alone, one graph layer then mixes sensors and space and time meet only once before the head. "
              "The other possible order applies the graph operator first and the temporal operator second, while the assignment fixes the temporal operator first. "
              f"The fixed graph model has {mc['fixed']['n_params']:,} trainable parameters, counted in code. "
              f"A code check found a largest difference of {mc['fixed']['graph_loop_max_diff']:.1e} between the batched graph layer and a plain loop over batch items. The notebook table rounds this value to 0.0. "
              "Training minimises the mean absolute error on normalised values over observed targets only, because MAE is less dominated than MSE by occasional traffic spikes. Metrics are computed after undoing the normalisation, over all four steps and all sensors. "
              f"The fixed graph model reaches a validation MAE of {mean_std(f_row, 'val_mae', n_seeds)} and RMSE of {mean_std(f_row, 'val_rmse', n_seeds)}. "
              f"It reaches a test MAE of {mean_std(f_row, 'test_mae', n_seeds)} and RMSE of {mean_std(f_row, 'test_rmse', n_seeds)} vehicles per fifteen minutes"
              + (f", given as mean ± standard deviation over seeds {seeds_text}. " if n_seeds > 1 else ". ")
              + cap_text + "\n\n")

    n_row = rows["none"]
    md.append(q_line("Q4(c) Temporal-Only Baseline. Implement a temporal-only model using the same temporal encoder and forecasting head, but without graph convolution. Compare its forecasting performance with your graph-based model. Explain whether explicitly modeling spatial dependencies improves forecasting accuracy."))
    md.append("The temporal only model uses the same encoder, the same weight $W_g$ with ReLU and the same head, but replaces $\\hat A$ with the identity, so each sensor sees only its own history. "
              "This keeps the parameter count and the training protocol identical, so the graph mixing is the only difference. "
              f"It reaches a validation MAE of {mean_std(n_row, 'val_mae', n_seeds)} and RMSE of {mean_std(n_row, 'val_rmse', n_seeds)}. "
              f"It reaches a test MAE of {mean_std(n_row, 'test_mae', n_seeds)} and RMSE of {mean_std(n_row, 'test_rmse', n_seeds)}. "
              + compare("fixed graph model", f_row, "temporal only model", n_row, "test_mae", n_seeds)
              + ("The validation split gives the same verdict, with a validation MAE of "
                 f"{f2(f_row['val_mae_mean'])} for the fixed graph model against {f2(n_row['val_mae_mean'])} for the temporal only model. "))
    gain = f_row["test_mae_mean"] < n_row["test_mae_mean"]
    md.append("Explicit spatial modelling therefore improves accuracy on this test period, likely because the neighbour features carry upstream and downstream changes that a sensor's own two hours do not show yet. "
              if gain else
              "Explicit spatial modelling with the fixed road graph does not improve accuracy on this test period, within this architecture and training budget. "
              f"The gap is already present on the training data, where the final normalised training MAE is {f_row['final_train_loss_mean']:.4f} for the fixed graph model against {n_row['final_train_loss_mean']:.4f} for the temporal only model, so the graph model underfits rather than overfits. ")
    if not gain and diag:
        md.append(f"In the layer $H=\\mathrm{{ReLU}}(\\hat A Z W_g)$ a sensor keeps only its own weight on the diagonal of $\\hat A$, which averages {diag['fixed']['self_weight_mean']:.3f} for the fixed graph, "
                  f"so on average about {100 * (1 - diag['fixed']['self_weight_mean']):.0f}% of a sensor's features comes from other sensors, spread over {f2(diag['fixed']['effective_neighbours'])} effective neighbours. "
                  f"Applied to the training mean flow of each sensor, the fixed graph moves a sensor's level by {f2(diag['fixed']['level_gap_mean'])} vehicles on average, which is {pct(diag['fixed']['level_gap_pct_of_mean_level'])} of the mean level of {f2(diag['mean_sensor_level'])}. "
                  "The measured level gap shows that road neighbours carry flow levels that differ from the sensor's own, so the shared head likely cannot recover each sensor's level from the averaged features. "
                  "Two causes are likely. A highway sensor and an exit sensor can be close on the road yet weakly coupled. Degree normalisation turns the graph layer into an average. Averaging counts mixes the levels of busy and quiet roads, while average speeds lie on a more similar scale. "
                  "The level gap above measures this mixing. "
                  "The assignment layer also uses one weight $W_g$ for a sensor and its neighbours, which is the cheaper but less expressive variant of a graph layer. "
                  "A separate self weight would likely let a sensor keep its own level, but this was not tested because the assignment fixes the layer. "
                  "The temporal only model avoids this mixing, because each sensor keeps its own features and the shared head maps them straight to that sensor's future. ")
    md.append(""
              + "The trade off is that the graph layer costs one dense multiplication by a 716 by 716 matrix per batch, while the temporal only model ignores every relation between sensors.\n\n")

    md.append("# Question 5. Fixed versus Learnable Adjacency Matrices\n\n")
    md.append(q_line("Q5(a) Fixed Adjacency. Use the original LargeST road-distance adjacency matrix. For a symmetric nonnegative graph, a common normalization is Â = D^(−1/2)(A + I)D^(−1/2), where D is the degree matrix of A + I. For a directed graph, choose and document an appropriate directed normalization instead of assuming symmetry. Train the forecasting architecture using this fixed adjacency matrix. The graph must remain unchanged throughout training. Report MAE, RMSE, and training time."))
    md.append("The fixed graph uses the row normalisation\n\n$$\\hat A=D_{\\text{out}}^{-1}(A+I),\\qquad (D_{\\text{out}})_{ii}=\\sum_j (A+I)_{ij},$$\n\n"
              "so each sensor takes a weighted average of itself and its out neighbours $N(i)$, with weights that add to one in every row. ")
    md.append(("This form is used instead of $D^{-1/2}(A+I)D^{-1/2}$ because the matrix is not symmetric while the symmetric form assumes $A_{ij}=A_{ji}$. "
               if not st["symmetric"] else "The matrix is symmetric, but the row form is still used so that the fixed and learned graphs are normalised the same way. ")
              + "Row normalisation also matches the neighbourhood $N(i)$ used in Question 3 and the row softmax of the learned graph in Q5(b), so the hybrid graph in Q5(c) mixes two matrices of the same kind. "
              "A graph layer can also be written for incoming edges, where node i averages over the nodes j with $A_{ji}>0$ and divides by the in degree. "
              "This report follows the assignment, where $\\hat A$ multiplies Z from the left and $N(i)=\\{j: A_{ij}\\neq 0\\}$, so each sensor averages over its out neighbours. "
              "The transposed graph, which would average over in neighbours, was not tested. "
              "The normalised matrix is stored as a fixed buffer outside the optimiser. Code asserts that it is unchanged after training. "
              "This model is the same run as the graph model in Q4(b), because Question 5 states that Question 4 used the physical road graph. "
              f"It reaches a test MAE of {mean_std(f_row, 'test_mae', n_seeds)}, a test RMSE of {mean_std(f_row, 'test_rmse', n_seeds)} "
              f"and a training time of {mean_std(f_row, 'train_s', n_seeds)} seconds on {device_text}, with an average run length of {f_row['epochs_mean']:g} epoch{'s' if f_row['epochs_mean'] != 1 else ''} at {f2(f_row['s_per_epoch_mean'])} seconds per epoch including one validation pass.\n\n")

    a_row = rows["adaptive"]
    md.append(q_line("Q5(b) Learnable Adjacency. Instead of using the provided graph, construct a graph whose relationships are learned during training. Introduce two trainable node embedding matrices E1, E2 ∈ RN×de, where de is the node embedding dimension. Construct an adaptive adjacency matrix Aadaptive = Softmax(ReLU(E1E2⊤)), with the softmax applied row-wise. The embedding matrices are initialized randomly and updated using backpropagation. Required: 1. Implement the learnable node embeddings. 2. Construct the adaptive adjacency matrix. 3. Train the same forecasting architecture using adaptive adjacency. 4. Visualize the learned adjacency matrix. 5. Compare its forecasting accuracy with fixed adjacency. For computational efficiency, you may retain the top k = 10 or 20 learned neighbors per node. If sparsification is used, explain whether it is applied before or after normalization and how gradients propagate."))
    md.append(f"Two trainable matrices $E_1,E_2\\in\\mathbb{{R}}^{{716\\times {cfg['d_e']}}}$ start from standard normal values under the fixed seed. "
              "The model computes $A_{\\text{adaptive}}=\\mathrm{softmax}(\\mathrm{ReLU}(E_1E_2^\\top))$ row by row in its own code, with no graph learning library. Code asserts that every entry is nonnegative and every row adds to one. "
              "The graph is rebuilt from the embeddings at every forward pass, so the forecasting loss reaches $E_1$ and $E_2$ through the softmax and the ReLU by ordinary backpropagation. "
              "This follows the Graph WaveNet construction (Wu et al., 2019), which uses two embeddings so that the learned graph can be directed. "
              f"With an embedding size of {cfg['d_e']}, far below 716, the matrix does not have full freedom, so this method avoids fitting all 716 × 716 links separately. "
              "No top k sparsification is used, because a dense 716 by 716 matrix is cheap at this size and keeps every gradient path open. "
              "The softmax after the ReLU turns every zero into the same small positive weight, so the learned graph is dense and its strongest links matter most.\n\n")
    md.append(figure(f"{FIG}/q5b_learned_adjacency.png", f"Figure 5. The plot shows the normalised fixed graph and the learned adaptive graph for seed {seeds[0]} on a log scale."))
    ga0 = graph[0]
    md.append(f"Figure 5 compares the two graphs on one log scale. The effective number of neighbours per row, $1/\\sum_j \\hat A_{{ij}}^2$, averages {f2(data['fixed_effective_neighbours'])} for the fixed graph and {f2(ga0['effective_neighbours'])} for the learned graph. "
              f"The strongest weight in each learned row averages {ga0['max_row_weight_mean']:.4f}, against a uniform weight of {ga0['uniform_weight']:.4f}. "
              f"The adaptive model reaches a test MAE of {mean_std(a_row, 'test_mae', n_seeds)} and RMSE of {mean_std(a_row, 'test_rmse', n_seeds)}. "
              + compare("adaptive graph model", a_row, "fixed graph model", f_row, "test_mae", n_seeds) + "\n\n")

    h_row = rows["hybrid"]
    md.append(q_line("Q5(c) Hybrid Adjacency. A hybrid graph combines the original road-network relationships with learned dependencies. One possible formulation is Ahybrid = αAfixed + (1 − α)Aadaptive, where α ∈ [0, 1]. Normalize both adjacency matrices consistently before combining them. You may use a fixed value such as α = 0.5, or learn α during training. Train the same forecasting architecture using the hybrid graph, and compare the performance of fixed, adaptive, and hybrid adjacency. All three experiments must use the same temporal encoder, forecasting head, dataset splits, and training protocol."))
    md.append(f"The hybrid graph uses a fixed $\\alpha={cfg['alpha']}$ with the row normalised fixed graph and the row softmax learned graph, so both terms have rows that add to one and so does their mix, which code asserts. "
              "All three graph models share the encoder, head, splits, optimiser, epochs, early stop rule and seeds. Only the adjacency differs. "
              "The convex mix keeps the total weight of each row at one, so the hybrid graph needs no extra normalisation. "
              f"The hybrid model reaches a test MAE of {mean_std(h_row, 'test_mae', n_seeds)} and RMSE of {mean_std(h_row, 'test_rmse', n_seeds)}. "
              + compare("hybrid graph model", h_row, "fixed graph model", f_row, "test_mae", n_seeds)
              + compare("hybrid graph model", h_row, "adaptive graph model", a_row, "test_mae", n_seeds)
              + (f"Every hybrid seed beats every adaptive seed, but the best adaptive seed at {f2(min(r['test']['mae'] for r in runs if r['graph'] == 'adaptive'))} is only {f2(min(r['test']['mae'] for r in runs if r['graph'] == 'adaptive') - max(r['test']['mae'] for r in runs if r['graph'] == 'hybrid'))} vehicles above the worst hybrid seed, so with three seeds this order is likely but not settled. "
                 if min(r['test']['mae'] for r in runs if r['graph'] == 'adaptive') > max(r['test']['mae'] for r in runs if r['graph'] == 'hybrid') else "The seed ranges of the hybrid and adaptive graph models overlap, so their order is not settled with three seeds. "))
    best = min(["fixed", "adaptive", "hybrid"], key=lambda g: rows[g]["test_mae_mean"])
    md.append(f"The {rows[best]['Adjacency'].lower()} graph has the lowest mean test MAE of the three. "
              "The trade off is that the learned and hybrid graphs add 716 × 2 × " + f"{cfg['d_e']} = {716 * 2 * cfg['d_e']:,} parameters and a dense softmax per step, while the fixed graph costs nothing to learn but can only use links that the road data already contains.\n\n")

    md.append("# Question 6. Experimental Analysis and Interpretation\n\n")
    md.append(q_line("Q6(a) Forecasting Comparison. Report your results using the following table, with columns Model, Adjacency, MAE, RMSE and Training time and rows Temporal-only with None, STGNN with Fixed, STGNN with Adaptive and STGNN with Hybrid. Include plots showing predicted versus actual traffic flow for at least three sensors."))
    md.append("Table 4. This table gives the test MAE and RMSE in vehicles per fifteen minutes and the training time in seconds for each model"
              + (f", as mean ± sample standard deviation over seeds {seeds_text}" if n_seeds > 1 else f", for seed {seeds[0]}")
              + ". Every model is scored on the same test period, April to December 2021.\n\n")
    md.append("| Model | Adjacency | MAE | RMSE | Training time |\n|---|---|---|---|---|\n")
    for g in ["none", "fixed", "adaptive", "hybrid"]:
        r = rows[g]
        md.append(f"| {r['Model']} | {r['Adjacency']} | {mean_std(r, 'test_mae', n_seeds)} | {mean_std(r, 'test_rmse', n_seeds)} | {mean_std(r, 'train_s', n_seeds)} s |\n")
    md.append("\n")
    md.append(figure(f"{FIG}/q6a_pred_vs_actual.png", f"Figure 6. The plot shows actual flow and predictions made {pred['step_ahead']} steps, one hour, ahead for three sensors from {pred['start']} to {pred['end']}."))
    overall = min(["none", "fixed", "adaptive", "hybrid"], key=lambda g: rows[g]["test_mae_mean"])
    md.append(f"Table 4 shows that the {rows[best]['Adjacency'].lower()} graph model has the lowest mean test MAE among the graph models"
              + (". " if overall != "none" else ", but the temporal only model has a lower test MAE than every graph model. ")
              + "The validation MAE gives the same order, with "
              + ", ".join(f"{f2(rows[g]['val_mae_mean'])} for the {prose_name(rows[g]['Model'] + ' ' + rows[g]['Adjacency'])}" for g in ["none", "fixed", "adaptive"])
              + f" and {f2(rows['hybrid']['val_mae_mean'])} for the hybrid graph model. "
              + f"Figure 6 compares the actual flow with predictions made one hour ahead by the temporal only model and the {rows[pred['best_graph']]['Adjacency'].lower()} graph model, which had the lowest mean validation MAE, for the first full test week. "
              + "Over this week and these three sensors the MAE is "
              + " and ".join(f"{f2(v)} vehicles for the {prose_name(name)}" for name, v in pred["week_mae"].items()) + ". "
              + "These three sensors were chosen as the most complete in 2020, so their week does not represent all 716 sensors. The full test MAE in Table 4 is the fair comparison. "
              + f"Training times in Table 4 differ mainly through run length, because every model needs between {f2(min(r['s_per_epoch_mean'] for r in rows.values()))} and {f2(max(r['s_per_epoch_mean'] for r in rows.values()))} seconds per epoch.\n\n")

    md.append(q_line("Q6(b) Graph Structure Analysis. Analyze the learned graph by answering: 1. Are the strongest learned connections also present in the original road-distance graph? 2. Do learned connections tend to occur between geographically nearby sensors? 3. Can the learned graph capture dependencies absent from the physical graph? 4. Why might a learned adjacency matrix perform better or worse than a fixed adjacency matrix? Support your analysis with at least one graph visualization or quantitative comparison."))
    agg = {k: sum(float("nan") if g[k] is None else g[k] for g in graph) / len(graph) for k in ["share_in_road_pct", "chance_in_road_pct", "share_in_road_either_pct",
                                                              "chance_in_road_either_pct", "median_km_learned", "median_km_road",
                                                              "median_km_random", "absent_from_road_pct", "corr_learned_absent", "corr_random"]}
    ex = graph[0]["strongest_absent_example"]
    md.append("Table 5. This table gives statistics of the ten strongest learned links of every sensor, "
              + (f"averaged over seeds {seeds_text}" if n_seeds > 1 else f"for seed {seeds[0]}") + ".\n\n")
    md.append("| Statistic | Value |\n|---|---|\n"
              f"| Learned links present in the road graph | {pct(agg['share_in_road_pct'])} |\n"
              f"| Chance rate for a random link | {pct(agg['chance_in_road_pct'])} |\n"
              f"| Present in either direction | {pct(agg['share_in_road_either_pct'])} (chance {pct(agg['chance_in_road_either_pct'])}) |\n"
              f"| Median distance of learned links | {f2(agg['median_km_learned'])} km |\n"
              f"| Median distance of road links | {f2(agg['median_km_road'])} km |\n"
              f"| Median distance of random pairs | {f2(agg['median_km_random'])} km |\n"
              f"| Learned links absent from the road graph | {pct(agg['absent_from_road_pct'])} |\n"
              f"| Mean training correlation, absent learned links | {agg['corr_learned_absent']:.3f} |\n"
              f"| Mean training correlation, random pairs | {agg['corr_random']:.3f} |\n\n")
    md.append(f"Table 5 answers the first question. {pct(agg['share_in_road_pct'])} of the strongest learned links are also road links, against {pct(agg['chance_in_road_pct'])} for random links, "
              + road_verdict(agg)
              + f"For the second question, learned links have a median length of {f2(agg['median_km_learned'])} km, against {f2(agg['median_km_road'])} km for road links and {f2(agg['median_km_random'])} km for random pairs, "
              + geography_verdict(agg)
              + f"For the third question, the answer is yes. {pct(agg['absent_from_road_pct'])} of the strongest learned links are absent from the road graph. Their training flows correlate at {agg['corr_learned_absent']:.3f} on average against {agg['corr_random']:.3f} for random pairs. ")
    if ex:
        md.append(f"For seed {seeds[0]} the strongest such link runs from sensor {ex['from_id']} to sensor {ex['to_id']}, which are {f2(ex['distance_km'])} km apart with a training correlation of {ex['train_corr']:.3f}. "
                  + "The strongest absent links of the other seeds are "
                  + " and ".join(f"{g['strongest_absent_example']['from_id']} to {g['strongest_absent_example']['to_id']} at {f2(g['strongest_absent_example']['distance_km'])} km with a correlation of {g['strongest_absent_example']['train_corr']:.3f}" for g in graph[1:]) + ". ")
    md.append(corr_verdict(agg, ex)
              + "The road distance cutoff never connects such sensors. "
              "For the fourth question, a learned graph can do better because a missing edge is a coupling the model can never learn. The learned graph is free to add the links that the road cutoff leaves out. "
              "It can do worse because a spurious edge adds noise and all 716 × 716 links come from 14,320 embedding values fitted to one year of correlated windows. "
              + gen_gap_sentence(rows)
              + compare("adaptive graph model", a_row, "fixed graph model", f_row, "test_mae", n_seeds)
              + (f"The learned graph moves a sensor's training level by {f2(diag['adaptive']['level_gap_mean'])} vehicles on average against {f2(diag['fixed']['level_gap_mean'])} for the fixed graph, which likely explains part of its lower error. " if diag else "")
              + "\n\n")

    md.append(q_line("Q6(c) Critical Discussion. Discuss the following: Does a lower forecasting error necessarily mean that the learned graph represents actual road connectivity? Why might a hybrid graph outperform either fixed or adaptive adjacency alone? What are the limitations of using a static graph for traffic forecasting, given that traffic dependencies may change throughout the day?"))
    md.append(f"A lower error does not show that the learned graph is the road network. Only {pct(agg['share_in_road_pct'])} of its strongest links are road links, against a chance rate of {pct(agg['chance_in_road_pct'])}. "
              + "The three seeds also learn different strongest absent links, " + join_ids([f"{g['strongest_absent_example']['from_id']} to {g['strongest_absent_example']['to_id']}" for g in graph]) + ", "
              + f"while their test MAE ranges only from {f2(min(r['test']['mae'] for r in runs if r['graph'] == 'adaptive'))} to {f2(max(r['test']['mae'] for r in runs if r['graph'] == 'adaptive'))}. "
              "Different graphs therefore give similar errors, so error alone cannot identify the true road connections. ")
    md.append("A hybrid graph can beat both parts because the road term supplies reliable local links from the start of training while the learned term adds links that the road cutoff misses. "
              + ("In this run the hybrid graph has the lowest mean test MAE of the three graphs. "
                 + ("It still has a higher test MAE than the temporal only model, so the benefit is relative to the other graphs only. " if rows["hybrid"]["test_mae_mean"] > n_row["test_mae_mean"] else "")
                 if best == "hybrid" else f"In this run the hybrid graph does not have the lowest mean test MAE, which is held by the {rows[best]['Adjacency'].lower()} graph, so the benefit was not observed at α = {cfg['alpha']}. ")
              + "A static graph uses the same weights at every hour, but traffic links change during the day. "
              "A sensor may follow its upstream neighbour closely in the morning peak, when queues spread. It may barely relate to it at night, when flow is free. "
              "One fixed matrix must average these regimes, so it is too strong at some hours and too weak at others. It also cannot react to incidents or closures that change routes for a few hours. "
              "Learned node embeddings fix the adjacency once after training, while graph attention computes content dependent weights that change with the inputs at every time step. "
              "The trade off is that a dynamic graph that changes with time of day or with the inputs would capture these shifts, but it costs more computation and is harder to train on one year of data.\n\n")

    md.append("# Question 7. Reproducibility and Reporting\n\n")
    md.append(q_line("Q7 Reproducibility and Reporting. Submit a concise technical report containing your implementation choices, training configuration, results, visualizations, and discussion. Your submission must include: Source code with instructions to reproduce the experiments. Model hyperparameters, training epochs, and random seed. The final comparison table. At least one adjacency visualization. Predicted versus actual traffic plots. Hardware specifications and measured training times."))
    md.append(f"The code repository on page 1 holds the source code and a README with install, data download and run steps for every experiment. "
              "It also holds the trained model files, so a rerun reloads them and recomputes the errors without training again. "
              f"Table 2 lists every hyperparameter, the epoch rule and the {'seeds ' + ', '.join(str(s) for s in seeds) + ', which are' if n_seeds > 1 else 'seed ' + str(seeds[0]) + ', which is'} set for Python, NumPy and PyTorch before each run. "
              "Table 4 is the final comparison table. Figures 3 and 5 show the fixed and learned adjacency. Figure 6 shows predicted against actual traffic. "
              f"All runs used {device_text} with {hw['cpu_threads']} CPU threads, PyTorch {hw['torch']} and Python {hw['python']}. "
              + cap_text
              + "Training times in Table 4 are wall clock seconds for the full training loop including one validation pass per epoch"
              + (", reported as mean ± standard deviation over the seeds. " if n_seeds > 1 else ", measured once per model, so they are not repeated. ")
              + rerun_sentence(previous, table)
              + "\n\n")
    md.append("## References\n\n"
              "Li, Q., Han, Z. and Wu, X. M. (2018). Deeper insights into graph convolutional networks for semi supervised learning. AAAI.\n\n"
              "Torch Spatiotemporal documentation, LargeST dataset page, listing the SD subset as District 11 with 716 nodes and 17,319 edges. https://torch-spatiotemporal.readthedocs.io\n\n"
              "Liu, X. et al. (2023). LargeST: A benchmark dataset for large scale traffic forecasting. NeurIPS Datasets and Benchmarks. https://github.com/liuxu77/LargeST\n\n"
              "Wu, Z. et al. (2019). Graph WaveNet for deep spatial temporal graph modeling. IJCAI.\n\n")
    return md


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="")
    parser.add_argument("--roll", default="")
    parser.add_argument("--repo", default="")
    parser.add_argument("--out", default="report.docx")
    build(parser.parse_args())
