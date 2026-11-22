# AI651 Assignment 3: Graph neural networks for traffic forecasting

Code for all seven questions of AI651 Deep Learning for Space, Time and Graphs, Assignment 3, Fall 2026. Questions 1 and 2 are small NumPy checks. Questions 3 to 7 train four forecasting models on the LargeST San Diego data.

## Files

| File | Purpose |
|---|---|
| `25280081_PA3_AI651_BR_SOLVED_V3.ipynb` | Runs every question in order and writes all figures, results and the report |
| `pa3_toy.py` | Question 1 and Question 2 checks and the Q2 heatmap |
| `pa3_data.py` | San Diego extraction, 15 minute resampling, missing values, splits, windows, adjacency |
| `pa3_models.py` | Temporal encoder, graph convolution, forecast head, fixed, adaptive and hybrid graphs |
| `pa3_train.py` | Seeding, masked MAE loss, training with early stopping, MAE and RMSE evaluation |
| `pa3_analysis.py` | Plots and the learned graph analysis for Question 6(b) |
| `pa3_pipeline.py` | One function per question that ties the modules together |
| `build_report.py` | Writes `report.docx` from the files in `results/` |
| `requirements.txt` | Exact package versions from the reported run |

## Reproduce

1. Open `25280081_PA3_AI651_BR_SOLVED_V3.ipynb` in Google Colab with a GPU runtime (Runtime, Change runtime type, T4 GPU).
2. Add two Colab secrets, `KAGGLE_USERNAME` and `KAGGLE_KEY`, from the legacy `kaggle.json` credentials on the Kaggle API settings page.
3. Run all cells. The notebook downloads `ca_meta.csv`, `ca_rn_adj.npy`, `ca_his_raw_2020.h5` and `ca_his_raw_2021.h5` from the Kaggle dataset `liuxu77/largest`, reloads the 12 trained models from `checkpoints/` and writes `report.docx`. The saved models recompute the validation and test errors without training again. To retrain from scratch, delete `checkpoints/` first. GPU kernels are not bit for bit deterministic, so a retrain with the same seeds reproduces the reported numbers within the seed spread rather than exactly.

Without Kaggle keys, place the four files in `data/` by hand. Newly trained models are also saved in `checkpoints/`, so a disconnect does not lose them.

## Settings

Seeds 0, 1 and 2 on GPU, or seed 0 alone on CPU. Hidden size 32, node embedding size 10, hybrid weight 0.5, Adam with learning rate 0.001, batch 64, at most 30 epochs with early stopping after 5 epochs without a validation gain. Splits are January to December 2020 for training, January to March 2021 for validation and April to December 2021 for testing.

## Data

LargeST, Liu et al., NeurIPS 2023 Datasets and Benchmarks, https://github.com/liuxu77/LargeST. The San Diego subset is District 11 in `ca_meta.csv`, as in the LargeST repository.
