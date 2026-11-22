# GFANet: GEV-Guided Frequency-Aware Network for Remote Sensing Change Detection

This repository provides the official implementation of **GFANet**, a GEV-Guided Frequency-Aware Network for remote sensing change detection.

> **Paper**: *GEV-Guided Frequency-Aware Network for Remote Sensing Change Detection*
> **Task**: Remote Sensing Change Detection
> **Keywords**: Change Detection, Pseudo-Change Suppression, GEV Distribution, Frequency-Aware Learning, DTCWT, Gaussian-Laplacian Pyramid

---

## Abstract

Remote sensing change detection (RSCD) aims to identify land-surface changes by comparing bi-temporal remote sensing images acquired over the same geographical area. Although deep learning methods have achieved remarkable progress, pseudo-change responses caused by non-semantic factors, such as illumination differences, seasonal variations, spatial misregistration, and background disturbances, remain a major challenge.

To address this issue, we propose **GFANet**, a **GEV-Guided Frequency-Aware Network** for remote sensing change detection. GFANet suppresses pseudo-change interference from three complementary perspectives. First, we design a **GEV-Guided Statistical Prior Modeling module (GSPM)**, which models local non-Gaussian responses using the Generalized Extreme Value (GEV) distribution and constructs a statistical prior map through a likelihood ratio test (LRT). Second, we propose a **Low-Frequency Consistency Correction module (LFCC)**, which extracts stable low-frequency structural information using the Dual-Tree Complex Wavelet Transform (DTCWT) and introduces a local Conditional Random Field (CRF) to enhance regional consistency. Third, we design a **Selective High-Frequency Enhancement module (SHFE)**, which employs a Gaussian-Laplacian pyramid and a Fourier amplitude-phase collaborative gating mechanism to selectively enhance reliable high-frequency residuals while suppressing pseudo high-frequency responses.

Extensive experiments on multiple public remote sensing change detection datasets demonstrate that GFANet achieves competitive performance against state-of-the-art methods.

---

## Framework

The overall architecture of GFANet consists of three key components:

### 1. GEV-Guided Statistical Prior Modeling Module

GSPM first performs superpixel segmentation to obtain approximately homogeneous local regions. It then models the pixel-intensity distribution within each region using the GEV distribution. Based on the likelihood ratio test, GSPM generates a region-level statistical prior map, which is further used to guide change feature enhancement.

### 2. Low-Frequency Consistency Correction Module

LFCC employs DTCWT to extract low-frequency structural representations with approximate shift invariance and directional selectivity. A local CRF is then constructed on the low-frequency feature space to impose structural consistency constraints, thereby reducing discontinuous predictions caused by region-level pseudo changes.

### 3. Selective High-Frequency Enhancement Module

SHFE uses a Gaussian-Laplacian pyramid to obtain multiscale high-frequency residuals. In the frequency domain, the amplitude spectrum is used to suppress spurious high-frequency responses, while the phase spectrum is used to highlight structural and boundary-related details. The enhanced residuals are progressively reconstructed to obtain the final high-frequency-enhanced representation.

---

## Main Contributions

* We propose **GSPM**, which uses the GEV distribution to characterize local non-Gaussian responses and generates a statistical prior map through LRT, reducing the sensitivity to appearance-induced pseudo changes.
* We propose **LFCC**, which extracts stable low-frequency structural information using DTCWT and improves regional consistency through a local CRF.
* We propose **SHFE**, which selectively enhances reliable high-frequency residuals through Gaussian-Laplacian pyramid decomposition and Fourier amplitude-phase collaborative gating.
* We integrate GSPM, LFCC, and SHFE into a unified remote sensing change detection framework, achieving strong performance on public RSCD benchmarks.

---

## Dataset

We conduct experiments on three public remote sensing change detection datasets:

| Dataset  | Task Type                       |                Image Size | Spatial Resolution | Description                               |
| -------- | ------------------------------- | ------------------------: | -----------------: | ----------------------------------------- |
| LEVIR-CD | Building change detection       |               1024 × 1024 |       ~0.5 m/pixel | Urban and suburban building changes       |
| WHU-CD   | Building change detection       | Large-scale aerial images |       ~0.2 m/pixel | High-resolution building change detection |
| JL1-CD   | Remote sensing change detection |                 512 × 512 |  ~0.5–0.75 m/pixel | Building and natural land-cover changes   |

---

## Installation

The code is implemented based on **PyTorch** and **Open-CD**.

### 1. Create environment

```bash
conda create -n gfanet python=3.8 -y
conda activate gfanet
```

### 2. Install PyTorch

Please install PyTorch according to your CUDA version. For example:

```bash
pip install torch torchvision torchaudio
```

### 3. Install Open-CD

Please refer to the official Open-CD installation instructions:

```bash
git clone https://github.com/likyoo/open-cd.git
cd open-cd
pip install -v -e .
```

---


## Acknowledgement

This project is developed based on PyTorch, Open-CD, and MMEngine. We sincerely thank the authors and contributors of these open-source projects.

---

## License

The license will be updated before the official release.

---
