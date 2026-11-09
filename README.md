# An SE-ResNet-26 Under 5M Parameters for CIFAR-10

NYU Tandon Deep Learning (Fall 2026), Project 1

**Team Punters:** Uvindu Salgado (uis202@nyu.edu) and Michael Wang (mzw2010@nyu.edu)
Kaggle competition: *NYU Tandon Deep Learning F26* (CIFAR-10 ResNet architecture challenge)
Report: [`report/report.pdf`](report/report.pdf)

Our final model is a ResNet with squeeze and excitation (SE) gates and **4,810,754 trainable parameters**. We trained
it from scratch on the 50,000 CIFAR-10 training images from the course, with no pretrained weights or outside data.

| Submission | Model | Parameters | Validation accuracy¹ | Kaggle public | Kaggle private |
|---|---|---|---|---|---|
| `final` | SE-ResNet-26, v4 recipe | 4,810,754 | 97.90% | 89.7% | **88.3%** |
| `final^2` | SE-ResNet-26, robust augmentation recipe | 4,810,754 | 97.82% | 89.6% | 88.0% |

**Final result: 88.3% private accuracy, 11th place.** We were 34th on the public leaderboard. We picked these two
submissions by validation accuracy instead of public score, and they ended up as our two best of ten on the private half.

¹ From the run with the same settings trained on 45,000 images. The submitted models were retrained on all 50,000.
All thirteen runs, including the ones that didn't work, are in [`results/runs.csv`](results/runs.csv) and in the report.

## What's in this repository

| Path | Contents |
|---|---|
| `results.ipynb` | **Start here.** Prints the parameter counts, recomputes validation accuracy from the saved weights, rebuilds both submitted files, and makes every figure in the report. |
| `train.py` | The training script exactly as we ran it on Kaggle for the final models. Flags cover every recipe we used (see below). |
| `models.py` | The model classes, copied from `train.py` so they can be imported. The notebook checks that the two match. |
| `checkpoints/` | Averaged (EMA) weights: `v4_full.pt` and `robust_full.pt` (the two submissions), `v4_val.pt` and `robust_val.pt` (the matching 45k runs), and `v2.pt`. |
| `kaggle_runs/` | The Kaggle notebook for every experiment with its full training output, numbered in the order we ran them. |
| `history/` | The exact script each experiment used. The code grew from `v1_train.py` into `train.py`. |
| `results/` | `runs.csv` (every run and its scores), training logs for each run, the submitted CSV files, test set probabilities, and the robustness and image statistics tables. |
| `figures/` | Figures made by `results.ipynb`, as PDF and PNG. |
| `report/` | LaTeX source and PDF of the report. |

## Quick start

```bash
git clone https://github.com/breeze215/cifar10-resnet-5m.git
cd cifar10-resnet-5m
pip install -r requirements.txt

# Competition data. You need a Kaggle account that has joined the competition; we don't include the images here.
kaggle competitions download -c cifar-10-res-net-architecture-challenge-dl
unzip cifar-10-res-net-architecture-challenge-dl.zip -d data
# data/ should now have train_labels.csv, test.csv, sample_submission.csv, train_images/ and test_images/

jupyter nbconvert --to notebook --execute --inplace results.ipynb   # about 15 min on a CPU, 1 min on a GPU
```

Among other things, the notebook prints:

```
FINAL MODEL (both submissions): SE-ResNet-26, trainable parameters = 4,810,754
v4_val      validation accuracy = 0.9790   (Kaggle log said 0.9790)
v4_full      rebuilt 2000 predictions, 2000 of 2000 match the uploaded file
robust_full  rebuilt 2000 predictions, 1999 of 2000 match the uploaded file
FINAL RESULT: private accuracy 0.883, rank 11 (we were 34 on the public leaderboard, so up 23 places)
```

## Reproducing the training runs

Every run used one NVIDIA T4 on Kaggle, mixed precision and batch size 128. We usually ran two at once, one on each
GPU of a "T4 x2" session. By default the scripts hold out 5,000 images (seed 0), the same ones behind every validation
number in the report. `--val_size 0` trains on all 50,000 images, which is what we did for submissions.

**Final submissions** (SE-ResNet-26 is `--blocks 5 4 3 --se`; about 32 seconds per epoch on a T4):

```bash
# "final" (v4), about 5.5 hours
python train.py --data_dir data --out_dir runs/v4_full --blocks 5 4 3 --se --epochs 600 --val_size 0
# "final^2" (robust), about 6.5 hours, since the extra augmentation runs on the CPU
python train.py --data_dir data --out_dir runs/robust_full --blocks 5 4 3 --se --aug robust --epochs 600 --val_size 0
```

Leave out `--val_size 0` to train the 45k versions (`v4_val`, `robust_val`). Adding
`--ref_models checkpoints/v4_val.pt checkpoints/v2.pt --ref_specs resnet-5-4-3-se resnet-4-4-3`
prints the robustness scorecard for those models next to the new one.

Each run saves `submission.csv`, `model_ema.pt`, `model.pt`, `train_log.csv` and `test_probs.npy` in `--out_dir`.
GPU training isn't bit for bit deterministic, so a rerun will land close to our logged numbers but not exactly on them.
The saved weights do reproduce our submissions: on a CPU the notebook matches 2,000 of 2,000 and 1,999 of 2,000 predictions.

**Every other experiment**, with the exact script we ran (the matching notebook in `kaggle_runs/` shows the output).
One note on names: the comments in `history/` call v2 "ResNet-26", but counting layers the way ResNet-18 is counted,
it has 24, and the final network has 26.

| # | Run | Command (add `--data_dir data`, except for the last row) | Notebook |
|---|---|---|---|
| 1 | v1 | `python history/v1_train.py --out_dir runs/v1 --epochs 200` | `01_v1_baseline` |
| 2 | v2 | `python history/v2_train.py --out_dir runs/v2 --epochs 300` | `02_v2_trivialaugment_cutmix_ema` |
| 3 | v2, all data | `python history/v3_train_full_sam.py --out_dir runs/v3_full --val_size 0 --epochs 300` | `03_v3_fulldata_and_sam` |
| 4 | SAM | `python history/v3_train_full_sam.py --out_dir runs/v3_sam --sam --epochs 200` | `03_v3_fulldata_and_sam` |
| 5 | AugMix | `python history/augmix_train.py --out_dir runs/augmix --epochs 300` | `04_augmix` |
| 6 | v2, all data, no grayscale | `python history/v2full_nogray_train.py --out_dir runs/v2full_nogray --epochs 300 --teachers --val_size 0` | `05_v2_fulldata_no_grayscale` |
| 7 | distillation | `python history/kd_train.py --out_dir runs/kd_val --epochs 200 --teachers <v2.pt> <v3_sam.pt>` | `06_distillation` |
| 8 | distillation, all data | `python history/kd_train.py --out_dir runs/kd_full --epochs 200 --val_size 0 --teachers <v2.pt> <v3_full.pt> <v3_sam.pt>` | `06_distillation` |
| 9, 10 | v4 | `python history/v4_train.py --out_dir runs/v4_val --blocks 5 4 3 --se --epochs 600` | `07_v4_se_resnet26_final` |
| 11, 12 | robust | `python history/v6_train_robust.py ...` (same as `train.py`) | `08_robust_aug_final2` |
| 13 | fine tuning | `python history/hardview_finetune.py --checkpoint checkpoints/v4_val.pt --data data` | `09_hardview_finetune` |

`train.py` is the last version of the script and can do everything the earlier ones did. With default flags it builds
the same model and data pipeline as `history/v4_train.py`, and `--teachers`, `--sam` and `--arch wrn` turn on
distillation, SAM and the large teacher (which we never trained).

## The recipe

**Model.** A 3x3 convolution with 64 channels, then three stages of 5, 4 and 3 residual blocks at 64, 128 and 256
channels (resolution 32, 16 and 8). Each block is conv, BN, ReLU, conv, BN, followed by an SE gate. The network ends
with global average pooling and a linear layer.

**Training.** SGD with Nesterov momentum 0.9 and batch size 128 for 600 epochs. The learning rate warms up to 0.1 over
5 epochs and then follows a cosine curve to zero. Weight decay is 5e-4, not applied to BN or bias terms. We use label
smoothing 0.1, CutMix on half the batches, and an exponential moving average of the weights (decay 0.9995) for
evaluation. Test predictions average 10 views of each image (the image and its mirror, each shifted by 2 pixels).

**Augmentation.** Random crop after padding 4 pixels, horizontal flip, then TrivialAugmentWide and
RandomGrayscale(0.05) for `final`. For `final^2`, a random choice between TrivialAugmentWide and AugMix, plus color
jitter, grayscale and blur done on the GPU.

## Use of the test images

We never had the test labels. We did look at simple pixel statistics of the unlabeled test images (brightness,
contrast, edge strength and how many are grayscale) to understand why validation accuracy wasn't carrying over to the
leaderboard. Section 9 of `results.ipynb` shows that analysis. One setting came from it: `RandomGrayscale(p=0.05)`,
used from v2 on. Run 6 is v2 trained without it, and it scored the same (89.8% public and 86.9% private, against
89.0% and 87.0% with it), so our results don't depend on that choice.

## References and acknowledgements

- Our residual block, CIFAR style first layer and baseline training settings follow the usual CIFAR ResNet
  conventions, as in [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) (MIT). We wrote our own code.
  Our layout (three stages, 64 base channels, about 4 blocks per stage) is close to Thakur, Chauhan and Gupta,
  *Efficient ResNets* ([arXiv:2306.12100](https://arxiv.org/abs/2306.12100)), a previous project in this course.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016. Hu et al., *Squeeze-and-Excitation Networks*,
  CVPR 2018. Müller and Hutter, *TrivialAugment*, ICCV 2021. Hendrycks et al., *AugMix*, ICLR 2020. Yun et al.,
  *CutMix*, ICCV 2019. Foret et al., *Sharpness-Aware Minimization*, ICLR 2021. Hinton et al., *Distilling the Knowledge
  in a Neural Network*, 2015. Beyer et al., *Knowledge distillation: A good teacher is patient and consistent*, CVPR 2022.
  Zagoruyko and Komodakis, *Wide Residual Networks*, BMVC 2016. Recht et al., *Do ImageNet Classifiers Generalize to
  ImageNet?*, ICML 2019. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
  The full list is in the report.

## License

MIT (see `LICENSE`). The CIFAR-10 images aren't included here and stay under the competition's rules.
