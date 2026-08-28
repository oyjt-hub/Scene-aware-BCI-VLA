# Scene-Aware SSVEP-BCI Controlled Robotic Manipulation Framework Integrated with Vision-Language-Action Models

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-red)](https://github.com/oyjt-hub/Scene-aware-BCI-VLA)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)

> **Official implementation of the paper:**  
> *"Scene-Aware SSVEP-BCI Controlled Robotic Manipulation Framework Integrated with Vision-Language-Action Models"*  
> **Authors:** Jun Song, Jietong Ouyang, Jason J. R. Liu, Hak-Keung Lam, Shuping He, Changyin Sun.

---

## 🌟 Overview

Deploying Vision-Language-Action (VLA) foundation models in assistive robotics faces a fundamental communication bottleneck with non-invasive Brain-Computer Interfaces (BCIs): continuous neural control induces severe user fatigue, while discrete BCI triggers lack semantic richness for dexterous tasks.

This repository provides an end-to-end **tri-level shared-control architecture**:
1. **Scene-Aware Visual Interface:** Uses **Florence-2** and **Grounded-SAM-2** to extract candidate targets and dynamically overlays multi-frequency SSVEP flickering masks directly on actionable objects.
2. **Context-Aware Reasoning Module:** Decodes sparse EEG selections via **FBCCA** and prompts an LLM (**Gemini-2.5-flash**) with scene context to infer precise manipulation instructions.
3. **VLA Execution Module:** Fine-tunes **$\pi_{0.5}$ (via OpenPI)** and **StarVLA** policies, integrated with an **Asynchronous Action Smoother** (EMA filtering, deadband filtering, and motion interpolation) for continuous dual-arm manipulation (e.g., cloth folding and fruit sorting).

<p align="center">
  <img src="assets/pipeline.png" width="95%" alt="System Pipeline"/>
</p>

### 🚀 Key Highlights
- **Ultra-Low User Effort:** Reduces Human Effort Ratio (HER) down to **4.0%**, completing tasks with only **1 to 3 discrete BCI triggers**.
- **Deformable Object Manipulation:** Expands BCI-robot manipulation from rigid pick-and-place to high-DoF deformable object tasks (e.g., dual-arm cloth folding).
- **High Success Rate:** Achieves **96.0%** success rate across 10 simulation tasks on the LIBERO-Goal benchmark.
- **Low Cognitive Workload:** Verified by NASA-TLX subjective evaluations (averaging between 1.4 and 3.6 across all dimensions).

---

## 🏗️ System Architecture & Submodules

The project integrates the following core components:
* **Perception:** [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) (Florence-2-large + SAM-2-hiera-large) for dynamic target extraction and visual mask rendering.
* **BCI Decoding:** 11-channel EEG acquisition (Neuracle NeuSenW) decoded via Filter Bank Canonical Correlation Analysis (FBCCA).
* **Cognitive Middleware:** Gemini-2.5-flash for intent disambiguation and task instruction synthesis.
* **Embodied Policy:** [OpenPI ($\pi_{0.5}$)](https://github.com/Physical-Intelligence/openpi) for real-world dual-arm execution & [StarVLA](https://github.com/starvla/starvla) for simulation.
* **Kinematic Layer:** Asynchronous producer-consumer control thread with receding horizon ($H=50$) and temporal Exponential Moving Average ($\beta=0.35$).

---

## 🛠️ Hardware Requirements

* **Dual-Arm Platform:** Agilex Piper 14-DoF dual-arm collaborative robot.
* **RGB-D Vision:** 3$\times$ Intel RealSense D435i cameras (1 overhead global view, 2 wrist-mounted views).
* **EEG Acquisition:** Neuracle NeuSenW wireless EEG system (Sampling rate: 1000 Hz; Channels: Fp1, Fp2, Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2).
* **Compute:** At least 48GB for inference and 100GB for fine-turning.

---

## 📦 Installation

### 1. Clone the repository and submodules
```bash
git clone --recursive https://github.com/oyjt-hub/Scene-aware-BCI-VLA.git
cd Scene-aware-BCI-VLA

# If you haven't cloned submodules recursively:
git submodule update --init --recursive
