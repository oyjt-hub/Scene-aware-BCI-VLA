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
3. **VLA Execution Module:** Fine-tunes **$\pi_{0.5}$ (via OpenPI)** policies, integrated with an **Asynchronous Action Smoother** (EMA filtering, deadband filtering, and motion interpolation) for continuous dual-arm manipulation (e.g., cloth folding and fruit sorting).

<p align="center">
<img width="2012" height="930" alt="image" src="https://github.com/user-attachments/assets/731df27f-6e1d-4537-b451-eaa0578b1a12" />
</p>

---

## 🏗️ System Architecture & Submodules

The project integrates the following core components:
* **Perception:** [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) (Florence-2-large + SAM-2-hiera-large) for dynamic target extraction and visual mask rendering.
* **BCI Decoding:** 11-channel EEG acquisition (Neuracle NeuSenW) decoded via Filter Bank Canonical Correlation Analysis (FBCCA).
* **Cognitive Middleware:** Gemini-2.5-flash for intent disambiguation and task instruction synthesis.
* **Embodied Policy:** [OpenPI](https://github.com/Physical-Intelligence/openpi) for real-world dual-arm execution & [StarVLA](https://github.com/starvla/starvla) for simulation.
* **Kinematic Layer:** Asynchronous producer-consumer control thread with receding horizon ($H=50$) and temporal Exponential Moving Average ($\beta=0.35$).

---

## 🛠️ Hardware Requirements

* **Dual-Arm Platform:** Agilex Piper 14-DoF dual-arm collaborative robot.
* **RGB-D Vision:** 3$\times$ Intel RealSense D435i cameras (1 overhead global view, 2 wrist-mounted views).
* **EEG Acquisition:** Neuracle NeuSenW wireless EEG system (Sampling rate: 1000 Hz; Channels: Fp1, Fp2, Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2).
* **Compute:** At least 48GB for inference and 100GB for fine-turning.

---
## 📦 Installation & Setup

### 1. Environment Preparation
We recommend using Conda to manage environment dependencies.

```bash
# Create and activate a clean conda environment
conda create -n bci-vla python=3.10 -y
conda activate bci-vla
```
# Install PyTorch with appropriate CUDA support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
### 2. Clone the main repository
```bash
git clone --recursive https://github.com/oyjt-hub/Scene-aware-BCI-VLA.git
cd Scene-aware-BCI-VLA
```

# Clone upstream dependencies
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git third_party/Grounded-SAM-2
git clone https://github.com/Physical-Intelligence/openpi.git third_party/openpi
```
# Clone upstream dependencies
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git third_party/Grounded-SAM-2
git clone https://github.com/Physical-Intelligence/openpi.git third_party/openpi

# Install Grounded-SAM-2
cd third_party/Grounded-SAM-2
pip install -e .
cd ../..

# Install OpenPI
cd third_party/openpi
pip install -e .
cd ../..

### 3. Replace the inference server script for remote VLA policy execution
cp src/vla_execution/serve_api.py third_party/openpi/serve_api.py

#  Deploy the robot execution client for Agilex Piper
cp src/robot_control/inference.py third_party/openpi/inference.py

### 4.Script Functionality Overview:
serve_api.py (VLA Inference Server): Hosts the fine-tuned π0.5 foundation policy on a dedicated GPU server. It continuously receives synchronized multi-camera RGB streams and natural-language prompts from the perception client, returning predicted action chunks.
inference.py (Robot Deployment Client): Executes on the local host machine connected to the Agilex Piper 14-DoF dual-arm robot. It handles real-time action consumption, temporal motion smoothing, deadband gripper filtering, and linear interpolation for continuous physical execution.

### 5. Hardware Configuration & EEG Customization
info
Adapting to Custom EEG Hardware:
In our paper, online EEG decoding was validated using a 9-channel Neuracle (NeuSenW) wireless acquisition system at 1000 Hz.
Please adapt or write the real-time EEG streaming interface to match your specific EEG hardware SDK.
