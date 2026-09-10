# Image Clustering with Visual and Textual Constraints on CIFAR-10

Unsupervised clustering of CIFAR-10 images using visual embeddings, caption-based text embeddings, and a fusion of both — evaluated against ground-truth labels with Cohen's Kappa. Built for the Machine Learning course "Clustering Assignment: Unsupervised Image Clustering."

## Overview

Clustering images by pixel/visual similarity alone often fails to separate semantically similar classes. This project tests whether adding *language* — auto-generated image captions — helps unsupervised clustering align better with true class labels, using a 10,000-image subset of CIFAR-10.

Pipeline:
1. **Visual features** — 512-d embeddings from an ImageNet-pretrained ResNet-18 (final classification layer removed).
2. **Textual features** — captions generated per-image with BLIP, then encoded into embeddings with Sentence-BERT.
3. **Fusion** — visual and textual embeddings standardized (`StandardScaler`) and concatenated.
4. **Clustering** — K-Means and Gaussian Mixture Models (K = 10) applied to each feature set.
5. **Evaluation** — each cluster is assigned the majority ground-truth label of its members, then compared to true labels via Cohen's Kappa Score.
6. **Visualization** — t-SNE projections of all three feature spaces, colored by ground truth vs. predicted cluster.

## Results

| Feature Type | K-Means (Cohen's Kappa) | GMM (Cohen's Kappa) |
|---|---|---|
| Visual (ResNet-18)        | 0.2099 | 0.2100 |
| Textual (BLIP + SBERT)    | 0.6523 | 0.6526 |
| Fused (Visual + Textual)  | 0.4844 | 0.4844 |

**Key finding:** caption-derived textual features substantially outperform raw visual features for this clustering task, and fusing the two lands in between — visual features add noise rather than complementary signal. Full discussion and t-SNE plots are in the notebook.

## Repository Structure

```
.
├── ImageClusteringCIFAR10.ipynb      # Main notebook: pipeline, results, discussion, conclusion
├── subset_indices.npy                # Indices defining the 10,000-image CIFAR-10 subset used
├── requirements.txt                  # Python dependencies
└── README.md
```

## Setup

```bash
git clone https://github.com/Vigilant-Tuhin/ml-img-clust-cifar10.git
cd ml-img-clust-cifar10
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended (BLIP captioning and ResNet feature extraction over 10,000 images are slow on CPU). The notebook auto-detects CUDA and falls back to CPU otherwise.

CIFAR-10 itself is **not** stored in this repo — the notebook downloads it automatically via `torchvision.datasets.CIFAR10(..., download=True)` on first run.

## Usage

Open and run `22MT30013_A3.ipynb` top to bottom (Jupyter or Colab). It will:
- download CIFAR-10 and load the subset defined by `subset_indices.npy`
- extract visual and textual features
- run K-Means/GMM clustering and print Cohen's Kappa for each feature type
- generate t-SNE visualizations, saved as PNGs alongside the notebook

## Dependencies

- `torch`, `torchvision`
- `transformers` (BLIP)
- `sentence-transformers`
- `scikit-learn`
- `numpy`, `matplotlib`, `tqdm`

See `requirements.txt` for pinned versions.

## Author

Tuhin Manik Biswas
