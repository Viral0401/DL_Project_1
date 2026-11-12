# Efficient ResNet for CIFAR-10 (≤ 5M parameters) — NYU Deep Learning Project 1

Team **Gulab jamun**: Viral Dalal, Aditya Patil, Shashwat Shah

A ResNet trained from scratch on CIFAR-10 under a 5-million-parameter limit, evaluated on a hidden Kaggle test set.
The full write-up is in `report/dl_project1_report.tex`.

## Final model
- ResNet, 3 stages of basic residual blocks (4 / 4 / 3 blocks, 64 / 128 / 256 channels), 64-channel stem
- **4,707,442 trainable parameters** (includes a 40-way rotation head used by SLA+SD)
- Training: SLA+SD rotation self-supervision, TrivialAugment, random crop + flip, 12px Cutout, label smoothing 0.1,
  Nesterov SGD, 5-epoch warm-up + cosine schedule to 1e-4, 200 epochs, flip test-time augmentation

## Selected submissions
| Model | Notebook | Public | Private |
|---|---|---|---|
| Try 4 — knowledge distillation (teachers: Try 2 + Try 3) | `final_pipeline/3_try4_kd_full.ipynb` | 91.5% | 86.5% |
| Try 8 — SWA continued from Try 4 | `final_pipeline/4_try8_swa.ipynb` | 91.4% | 86.7% |

## Reproducing the final models (run in this order)
1. `final_pipeline/1_try2_sla_val.ipynb` — Try 2 (45k train / 5k validation, 97.88% val, 98.02% with flip). Teacher 1.
2. `final_pipeline/2_try3_sla_full_ema.ipynb` — Try 3 (same recipe + EMA, all 50k images). Teacher 2.
3. `final_pipeline/3_try4_kd_full.ipynb` — Try 4: distills from the `final_resnet.pth` of Try 2 and Try 3.
   On Kaggle this run was interrupted twice by the platform and resumed from its per-epoch checkpoint
   (`RESUME_FROM`); training is identical to an uninterrupted run.
4. `final_pipeline/4_try8_swa.ipynb` — Try 8: starts from Try 4's final weights, trains 40 more epochs at a constant
   learning rate of 0.01 and averages 35 snapshots (SWA). The last cell recomputes BatchNorm statistics on augmented,
   4-way-rotated training batches; this cell produced the submitted predictions.

Notebooks 1–3 ran on Kaggle (2× T4); notebook 4 and Tries 6–7 ran on Google Colab (A100). Each notebook prints the
parameter count and its final results. Model weights are not stored in this repository (size); they are produced by
running the notebooks in order.

## Other experiments (`experiments/`)
| Notebook | Idea | Public | Private |
|---|---|---|---|
| `0_baseline.ipynb` | ResNet 3/3/3, 4.31M params, 60 epochs | 85.3% | 81.2% |
| `1_improved_v1.ipynb` | 4/4/3, RandAugment + MixUp (stopped at epoch 161/200) | 85.2% | 81.7% |
| `try5_kd3.ipynb` | distillation from 3 teachers, 180 epochs (epochs 145–180 finished on Colab) | 91.0% | 87.1% |
| `try6_kd4.ipynb` | distillation from 4 teachers + mirrored teacher targets | 89.6% | 86.8% |
| `try7_kd2_sam.ipynb` | Try 4 recipe + SAM (rho 0.05) | 90.8% | 87.1% |

Try 3 itself scored 90.2% public / 87.7% private.

## Requirements
Python 3, PyTorch, torchvision, numpy, pandas, scikit-learn, Pillow, tqdm, matplotlib.
Competition data must be attached on Kaggle (or downloaded) as described at the top of each notebook.
