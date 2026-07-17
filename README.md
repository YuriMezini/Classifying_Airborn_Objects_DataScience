# Classifying_Airborn_Objects_DataScience

# Classifying Airborne Objects: Drone vs. Bird Detection

Detecting consumer FPV drones near sensitive airspace (airports, military
installations) is a safety and security problem. The catch is that many
"drone" sightings are actually **birds**. This project builds and compares
classifiers/detectors that tell **drones apart from birds** in aerial imagery,
using both **RGB (color)** and **IR (thermal)** sensors.

**Authors:** Benjamin Braiman, Jurgen Mezinaj, Ganiyat Shodunke, Kushal Bhowmick

*Erdős Institute — Data Science Boot Camp (Summer 2026). Group project.*

---

## Overview

We treat the problem as small-target object detection and work through a full
pipeline: cleaning the raw data into a single manifest, building a **non-neural
statistical baseline**, training **YOLOv8n** detectors on RGB, IR, and an
**RGB+IR early-fusion** input, and then stress-testing the models for
**robustness** and **inference speed**. We also ran two smaller follow-up
experiments: a **Faster R-CNN (Detectron2)** detector and a **Non-Local Means
denoising** study.

The full write-up is in [`Erdos-Project-July26/main.pdf`](Erdos-Project-July26/main.pdf).

---

## Dataset

### Which database we chose, and why

We use the **IEEE SPS VIP Cup 2025 dataset** (accessed through a Kaggle mirror).
It is a **real** collection of aerial imagery captured specifically to tell
small drones apart from birds — not a synthetic or pre-cleaned Kaggle set, which
matters for the Erdős "real data + value add" requirement. We chose it for three
reasons:

1. **It is genuinely hard.** At long range, drones and birds both appear as tiny
   dark shapes against a bright sky, often only a handful of pixels across. That
   resemblance is the core challenge and is exactly why a naive feature-based
   classifier struggles.
2. **It is dual-modality.** Every scene is captured **simultaneously in RGB
   (color) and IR (thermal)**, co-registered frame-for-frame. This let us
   compare modalities and build an RGB+IR fusion model.
3. **It needed real cleaning.** The raw archive is large and messy, so getting
   it into a usable, reproducible form was a substantial part of the work.

- **Task:** detect/classify two classes — `0 = bird`, `1 = drone`.
- **Labels:** YOLO format — one line per object (class ID + normalized
  bounding box: center *x*, center *y*, width, height).

### Cleaning and building a single manifest

The raw archive holds roughly **230,000 files**, a large fraction of which are
macOS junk (`__MACOSX/`, `._*` resource forks, `.DS_Store`). After filtering, we
located **115,160 clean images** and paired each with its label file by
mirroring the `images`/`labels` directory structure — with **zero pairing
failures**. Split and modality were inferred from each file's path, and the whole
inventory was saved as a **single manifest CSV** so every downstream notebook
loads the manifest instead of re-walking the archive (important because the work
ran in Google Colab, where sessions disconnect often).

| Split | RGB images | IR images |
|-------|-----------|-----------|
| train | 40,306 | 40,306 |
| val   | 8,637  | 8,637  |
| test  | 8,637  | 8,637  |
| **total** | **57,580** | **57,580** |

**RGB/IR pairing for fusion:** RGB and IR frames match by filename stem with
**100% overlap in every split** (57,580 matched pairs); a spot-check of 300
random pairs found 0 label mismatches.

### Class balance and label convention

`data.yaml` names the classes only generically (`class_0`, `class_1`), so we
tallied object instances directly and cross-referenced the totals against a
published paper using the same VIP Cup data (~30k bird / ~36k drone per
modality). This strongly indicates **class 0 = bird, class 1 = drone**, which we
adopt throughout (and list as an inferred convention in the limitations).

| Split | Bird (class 0) | Drone (class 1) |
|-------|----------------|-----------------|
| train | 19,931 | 25,606 |
| val   | 4,246  | 5,466  |
| test  | 4,049  | 5,598  |
| **total** | **56,452** | **73,340** |

Drones outnumber birds by roughly 1.3 : 1.

> The dataset itself is **not** committed to this repo (size/licensing). See
> [Reproducing the results](#reproducing-the-results) for how to point the
> notebooks at your own copy.

---

## Models

1. **Model 1 — Non-neural statistical baseline.** Laplacian-energy proposal
   generation → statistical chip features (size, aspect ratio, color/intensity
   statistics, Laplacian energy) → **Gaussian Naive Bayes** classifier.
2. **Model 2a / 2b — YOLOv8n** trained on RGB and on IR separately.
3. **Model 2c — YOLOv8n with RGB + IR early fusion.**
4. **Additional — Faster R-CNN (ResNet-50 FPN) via Detectron2** *(preliminary)*.

---

## Results

Detection accuracy is reported as **mAP@0.5** (there is no standard accuracy
scalar for detection). The baseline row reports classification accuracy on
object chips.

| Model | Modality | mAP@0.5 | Precision | Recall | F1 |
|-------|----------|---------|-----------|--------|-----|
| Baseline (Model 1 + Gaussian NB) | RGB | 0.390\* | 0.321 | 0.275 | 0.296 |
| YOLOv8n | IR | 0.745 | 0.815 | 0.723 | 0.763 |
| **YOLOv8n** | **RGB** | **0.820** | **0.889** | **0.759** | **0.812** |
| YOLOv8n (fusion) | RGB+IR | 0.788 | 0.827 | 0.723 | 0.754 |

\* accuracy, not mAP. YOLO numbers are from a quick-test subset run (see
[Limitations](#limitations)).

**Takeaway:** the neural detectors beat the statistical baseline by a wide
margin, and RGB is the single strongest modality on clean data.

### Robustness (mAP@0.5 under corruption)

Models were tested under Gaussian blur, sensor noise, brightness shift, and JPEG
compression.

| Modality | Clean | Worst case | Relative drop |
|----------|-------|-----------|---------------|
| IR  | 0.729 | 0.405 | −44.4% |
| RGB | 0.826 | 0.326 | −60.5% |

RGB is more accurate on clean data but degrades more under corruption; IR loses
less accuracy in the worst case.

### Inference speed (CUDA)

| Modality | ms / image | FPS |
|----------|-----------|-----|
| RGB | 9.7  | ~103 |
| IR  | 10.9 | ~92  |

Both models run comfortably in real time (~5.9 MB weights, ~3.0M parameters).

### Additional experiments

- **Faster R-CNN (Detectron2):** a short 300-iteration fine-tune from
  COCO-pretrained weights. Preliminary COCO scores AP@0.5 ≈ 16.5 — low as
  expected for such a short run; it confirms the pipeline works and is ready for
  full training before any fair comparison to YOLOv8.
- **Non-Local Means denoising:** explored to suppress sky shot-noise. At the
  strength tested it over-smooths fine detail, which is risky for few-pixel
  targets — flagged as future work rather than an improvement.

---

## Repository structure

```
.
├── README.md
├── 00_setup_and_data.ipynb          # data loading + manifest construction
├── 01_baseline_statistical.ipynb    # Model 1: proposals + Gaussian NB
├── 02_model_rgb.ipynb               # YOLOv8n on RGB
├── 03_model_ir.ipynb                # YOLOv8n on IR
├── 04_model_fusion.ipynb            # YOLOv8n RGB+IR early fusion
├── 05_robustness_testing.ipynb      # corruption robustness
├── 06_speed_benchmarking.ipynb      # latency / FPS
├── 07_demo.ipynb                    # qualitative demo images
├── 08_report.ipynb                  # aggregates result tables/CSVs
├── UAV_classification.ipynb         # Faster R-CNN (Detectron2) + denoising
├── vip_cup_project/                 # results, manifests, figures, demo images
│   ├── manifests/                   # image/label manifest, class balance
│   ├── features/                    # baseline chip features
│   ├── results/                     # model_comparison, robustness, speed, plots
│   ├── report_exports/              # summary CSVs used in the report
│   └── demo/                        # annotated RGB/IR demo frames
└── Erdos-Project-July26/            # LaTeX report + slides
    ├── main.tex / main.pdf          # the written report
    ├── figures/                     # all report figures
    └── slides/                      # presentation slides
```

---

## Reproducing the results

The notebooks were developed in **Google Colab** with data on Google Drive.

1. **Environment.** Python 3.10+, plus:
   ```bash
   pip install ultralytics scikit-learn opencv-python pandas numpy matplotlib pillow
   # Faster R-CNN experiment only:
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```
2. **Data.** Place the drone/bird dataset (YOLO format, RGB + IR) on Drive and
   update the path at the top of `00_setup_and_data.ipynb`.
3. **Run in order:** `00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`.
   Each notebook writes its outputs (CSVs, plots, weights) under
   `vip_cup_project/`.
4. **Report.** Build the PDF:
   ```bash
   cd Erdos-Project-July26
   pdflatex main.tex && pdflatex main.tex
   ```

---

## Limitations

- Class mapping (`0 = bird`, `1 = drone`) was inferred from instance counts
  cross-referenced with a published paper; a visual spot-check is pending.
- `mAP@0.5` is used as the accuracy proxy for detection.
- The baseline is matched by center-hit rather than strict IoU (proposals bound
  objects loosely).
- YOLO numbers come from a **quick-test subset**; full 30-epoch training on the
  complete dataset is the immediate next step.
- Faster R-CNN training and denoising integration are preliminary.

---

## References

**Dataset**

- IEEE Signal Processing Society. *VIP Cup 2025: Drone vs. Bird Detection
  Challenge Dataset.* 2025. Real RGB and co-registered infrared aerial imagery
  with YOLO-format annotations; accessed through a Kaggle mirror.

**Primary reference**

- L. Yang, R. Ma, and A. Zakhor. "Drone Object Detection Using RGB/IR Fusion."
  *CoRR*, vol. abs/2201.03786, 2022.
  <https://arxiv.org/abs/2201.03786>

Additional background material included with the project covers YOLO (You Only
Look Once), air-to-air micro-UAV detection, HOG+SVM UAV detection, and reviews of
machine learning for drone detection.

---

*Completed for the Erdős Institute Data Science Boot Camp (Summer 2026).*
