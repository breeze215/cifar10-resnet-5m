# SE-ResNet-26 under 5M parameters — NYU Tandon Deep Learning (Fall 2026), Project 1

**Team Punters:** Uvindu Salgado (uis202@nyu.edu) · Michael Wang (mzw2010@nyu.edu)
Kaggle competition: *NYU Tandon Deep Learning F26* (CIFAR-10 ResNet architecture challenge).
Report: [`report/report.pdf`](report/report.pdf)

A ResNet with squeeze-and-excitation (SE) gates, **4,810,754 trainable parameters**, trained from scratch on the
50,000 CIFAR-10 training images supplied by the course, with no pretrained weights or external data.

| Final submission | Model | Params | Val acc. (5k held-out)¹ | Kaggle public | Kaggle private |
|---|---|---|---|---|---|
| `final` | SE-ResNet-26, v4 recipe | 4,810,754 | 97.90% | 89.7% | *pending* |
| `final^2` | SE-ResNet-26, robust-augmentation recipe | 4,810,754 | 97.82% | 89.6% | *pending* |

¹ Measured on the twin run trained on 45k images (same recipe); the submitted models were retrained on all 50k.
Every run, including the ones that did not work, is listed in [`results/runs.csv`](results/runs.csv) and discussed in the report.

## What is in this repository

| Path | Contents |
|---|---|
| `results.ipynb` | **Start here.** Executed notebook: parameter counts, validation accuracy recomputed from the checkpoints, regeneration of both submitted files, all report figures, the robustness scorecard, train/test shift analysis, leaderboard-noise analysis. |
| `train.py` | The training script, exactly as run on Kaggle for the final models. One file covers every recipe we used (flags below). |
| `models.py` | The model classes, copied verbatim from `train.py` so they can be imported (the notebook checks they are identical). |
| `checkpoints/` | EMA weights: `v4_full.pt`, `robust_full.pt` (the two submissions), `v4_val.pt`, `robust_val.pt` (45k twins used for validation), `v2.pt` (baseline). |
| `kaggle_runs/` | The executed Kaggle notebooks for every experiment, numbered in the order we ran them, with their full training output. |
| `history/` | The exact script each experiment used (the code evolved from `v1_train.py` to `train.py`). |
| `results/` | `runs.csv` (every run and its scores), per-run `train_log.csv` and stdout logs, submitted CSVs, test-set probabilities, scorecard and shift statistics. |
| `figures/` | Figures produced by `results.ipynb` (PDF and PNG). |
| `report/` | LaTeX source and PDF of the report. |

## Quick start

```bash
git clone https://github.com/breeze215/cifar10-resnet-5m.git
cd cifar10-resnet-5m
pip install -r requirements.txt

# Competition data (needs a Kaggle account that joined the competition). It is not redistributed here.
kaggle competitions download -c cifar-10-res-net-architecture-challenge-dl
unzip cifar-10-res-net-architecture-challenge-dl.zip -d data
# data/ should now contain train_labels.csv, test.csv, sample_submission.csv, train_images/, test_images/

jupyter nbconvert --to notebook --execute --inplace results.ipynb   # ~15 min on CPU, ~1 min on a GPU
```

The notebook prints, among other things:

```
FINAL MODEL (both submissions): SE-ResNet-26, trainable parameters = 4,810,754
v4_val      val accuracy (flip-averaged, EMA weights) = 0.9790   [logged on Kaggle GPU: 0.9790]
v4_full      regenerated 2000 predictions | agreement with uploaded file: 1.0000 (0 of 2000 differ ...)
robust_full  regenerated 2000 predictions | agreement with uploaded file: 0.9995 (1 of 2000 differ ...)
```

## Reproducing the training runs

All runs used one NVIDIA T4 (Kaggle), mixed precision, batch size 128. Two runs were usually launched at once,
one per GPU of a Kaggle "T4 x2" session. `--val_size 5000` (default) holds out the seed-0 split used for every
validation number in the report; `--val_size 0` trains on all 50,000 images (used for submissions).

**Final submissions** (SE-ResNet-26: `--blocks 5 4 3 --se`, ~32 s/epoch on a T4):

```bash
# "final"   (v4):     ~5.5 h
python train.py --data_dir data --out_dir runs/v4_full --blocks 5 4 3 --se --epochs 600 --val_size 0
# "final^2" (robust): ~6.5 h (augmentation runs on the CPU workers)
python train.py --data_dir data --out_dir runs/robust_full --blocks 5 4 3 --se --aug robust --epochs 600 --val_size 0
```

Drop `--val_size 0` to train the 45k twins (`v4_val`, `robust_val`). Add
`--ref_models checkpoints/v4_val.pt checkpoints/v2.pt --ref_specs resnet-5-4-3-se resnet-4-4-3`
to print the robustness scorecard for reference models next to the new one.

Each run writes `submission.csv`, `model_ema.pt`, `model.pt`, `train_log.csv` and `test_probs.npy` to `--out_dir`.
GPU kernels are not bit-deterministic, so reruns land close to, not exactly on, the logged numbers
(on CPU, the notebook's regenerated predictions match our uploads on 2,000/2,000 and 1,999/2,000 test images).

**Every other experiment** (exact scripts as run; `kaggle_runs/` shows the executed notebooks). Naming note: the
docstrings in `history/` call the v2 network "ResNet-26"; counting main-path conv layers plus the classifier, as in
"ResNet-18", it has 24 layers, and the final network has 26. The report uses the corrected names.

| Run | Command (add `--data_dir data`, except the last row) | Notebook |
|---|---|---|
| v1 | `python history/v1_train.py --out_dir runs/v1 --epochs 200` | `01_v1_baseline` |
| v2 | `python history/v2_train.py --out_dir runs/v2 --epochs 300` | `02_v2_trivialaugment_cutmix_ema` |
| v3 full data | `python history/v3_train_full_sam.py --out_dir runs/v3_full --val_size 0 --epochs 300` | `03_v3_fulldata_and_sam` |
| v3 SAM | `python history/v3_train_full_sam.py --out_dir runs/v3_sam --sam --epochs 200` | `03_v3_fulldata_and_sam` |
| AugMix (MW) | `python history/mw_augmix_train.py --out_dir runs/mw_augmix --epochs 300` | `04_mw_augmix` |
| v2 on 50k, no grayscale (MW) | `python history/mw_v2full_train.py --out_dir runs/mw_v2full --epochs 300 --teachers --val_size 0` | `05_mw_v2_fulldata` |
| distillation (val) | `python history/kd_train.py --out_dir runs/kd_val --epochs 200 --teachers <v2.pt> <v3_sam.pt>` | `06_distillation` |
| distillation (50k) | `python history/kd_train.py --out_dir runs/kd_full --epochs 200 --val_size 0 --teachers <v2.pt> <v3_full.pt> <v3_sam.pt>` | `06_distillation` |
| v4 | `python history/v4_train.py --out_dir runs/v4_val --blocks 5 4 3 --se --epochs 600` | `07_v4_se_resnet28_FINAL` |
| robust | `python history/v6_train_robust.py ...` (same as `train.py`) | `08_v6_robust_aug_FINAL2` |
| hard-view fine-tune | `python history/hardview_finetune.py --checkpoint checkpoints/v4_val.pt --data data` | `09_hardview_finetune` |

`train.py` is the last version of the script and a superset of the earlier ones: with default flags it builds the
same model and data pipeline as `history/v4_train.py`, and `--teachers`, `--sam` and `--arch wrn` reproduce the
distillation, SAM and (unused) large-teacher options.

## The recipe in one paragraph

SE-ResNet-26: a 3x3 stem (64 channels) followed by 5/4/3 BasicBlocks (conv-BN-ReLU-conv-BN, then an SE gate)
at 64/128/256 channels and resolutions 32/16/8, global average pooling and a linear classifier. Training: SGD with
Nesterov momentum 0.9, learning rate 0.1 with a 5-epoch linear warm-up then cosine decay, weight decay 5e-4
(not on BN/bias), batch 128, 600 epochs, label smoothing 0.1, CutMix on half the batches, an exponential moving
average of the weights (decay 0.9995) used for evaluation, and 10-view test-time augmentation (2 flips x 5 shifts).
Augmentation: reflect-pad-4 random crop, horizontal flip, then TrivialAugmentWide + RandomGrayscale(0.05) (`final`),
or TrivialAugmentWide/AugMix plus GPU colour jitter, grayscale and blur (`final^2`).

## Use of the evaluation images

The test labels were never used. We did look at aggregate pixel statistics of the unlabeled test images
(brightness, contrast, edge energy, fraction of grayscale images) to understand why validation accuracy did not
transfer to the leaderboard; `results.ipynb` §9 shows the analysis. One design choice, `RandomGrayscale(p=0.05)` in
v2 onward, was motivated by it. Michael's re-run of v2 without it (`mw_v2full`) scored the same on the leaderboard
(0.898 vs 0.899), so our results do not depend on it.

## References and acknowledgements

- Residual block, CIFAR-style stem and baseline training settings follow the standard CIFAR ResNet conventions
  as implemented in [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) (MIT); our code was written
  independently. Our final layout (three stages, 64 base channels, [4,4,3]-style blocks) is close to
  Thakur, Chauhan & Gupta, *Efficient ResNets* ([arXiv:2306.12100](https://arxiv.org/abs/2306.12100), a previous project in this course).
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 · Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018 ·
  Müller & Hutter, *TrivialAugment*, ICCV 2021 · Hendrycks et al., *AugMix*, ICLR 2020 · Yun et al., *CutMix*, ICCV 2019 ·
  Foret et al., *Sharpness-Aware Minimization*, ICLR 2021 · Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015 ·
  Beyer et al., *Knowledge distillation: A good teacher is patient and consistent*, CVPR 2022 · Zagoruyko & Komodakis, *Wide Residual Networks*, BMVC 2016 ·
  Recht et al., *Do ImageNet Classifiers Generalize to ImageNet?*, ICML 2019 (CIFAR-10.1) · Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
  Full list in the report.

## License

MIT (see `LICENSE`). The CIFAR-10 images are not included and remain subject to the competition rules.
