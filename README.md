# Scene-Aware SSVEP-BCI Controlled Robotic Manipulation Framework Integrated with Vision-Language-Action Models

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-red)](https://github.com/oyjt-hub/Scene-aware-BCI-VLA)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)

> **Official implementation of the paper:**
> *"Scene-Aware SSVEP-BCI Controlled Robotic Manipulation Framework Integrated with Vision-Language-Action Models"*
> **Authors:** Jun Song, Jietong Ouyang, Jason J. R. Liu, Hak-Keung Lam, Shuping He, Changyin Sun.

[![Demo](https://img.shields.io/badge/Demo-Bilibili-FB7299?logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1YaYH6iESB/?spm_id_from=333.1387.homepage.video_card.click&vd_source=628cfed84bbf2057bb65b782c484b8b6)

## 🌟 Overview

Deploying Vision-Language-Action (VLA) foundation models in assistive robotics faces a fundamental communication bottleneck with non-invasive Brain-Computer Interfaces (BCIs): continuous neural control induces severe user fatigue, while discrete BCI triggers lack semantic richness for dexterous tasks.

This repository provides an end-to-end **tri-level shared-control architecture**:

1. **Scene-Aware Visual Interface:** Uses **Florence-2** and **Grounded-SAM-2** to extract candidate targets and dynamically overlays multi-frequency SSVEP flickering masks directly on actionable objects.
2. **Context-Aware Reasoning Module:** Decodes sparse EEG selections via **FBCCA** and prompts an LLM (**Gemini-2.5-flash**) with scene context to infer precise manipulation instructions.
3. **VLA Execution Module:** Fine-tunes **π0.5** policies, integrated with an **Asynchronous Action Smoother** (EMA filtering, deadband filtering, and motion interpolation) for continuous dual-arm manipulation.

<p align="center">
<img width="2012" height="930" alt="architecture" src="https://github.com/user-attachments/assets/731df27f-6e1d-4537-b451-eaa0578b1a12" />
</p>

---

## 📁 Repository Structure

The repository is organized into **four parts**: the project's own BCI decoding module, the two upstream base frameworks it builds on, and the system-integration scripts.

```text
.
├── Scene-aware-BCI-VLA/     # ① BCI decoding module (this project's core)
│   ├── triggerBox.py        #    SSVEP flicker stimulation & trigger-box synchronization
│   ├── fbcca.py             #    Filter-Bank Canonical Correlation Analysis (FBCCA) decoder
│   ├── interface.py         #    Neuracle NeuSenW wireless EEG acquisition interface
│   └── creat_raw_data.py    #    Raw EEG dataset construction
│
├── openpi/                  # ② VLA policy base — π0.5 fine-tuning & serving (modified fork)
│
├── Grounded-SAM-2/          # ③ Scene perception base — target extraction & SSVEP mask rendering (modified fork)
│
└── scripts/                 # ④ System integration, robot client & utilities
    ├── inference.py         #    Main entry — closed-loop BCI-VLA inference on the robot host
    ├── server_api.py        #    π0.5 policy server hosted on the GPU workstation
    ├── robot/               #    Robot-side execution clients (RTC_inference, SSVEP_inference, ...)
    ├── bci/                 #    Robot-side EEG streaming & decoding utilities
    ├── training/            #    Policy fine-tuning & normalization-statistics scripts
    ├── data_tools/          #    Dataset conversion (LeRobot / ROS)
    └── docker/              #    Dockerfiles & setup scripts for policy serving
```

> **Note:** `openpi/` and `Grounded-SAM-2/` are included **in full** so the repository is self-contained — no separate cloning of the upstream projects is required. Both keep their original internal structure, with our project-specific modifications on top.

---

## 🏗️ System Architecture

The project integrates the following core components:

* **Perception:** [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) (Florence-2-large + SAM-2-hiera-large) for dynamic target extraction and visual mask rendering.
* **BCI Decoding:** 11-channel EEG acquisition (Neuracle NeuSenW) decoded via Filter Bank Canonical Correlation Analysis (FBCCA).
* **Cognitive Middleware:** Gemini-2.5-flash for intent disambiguation and task instruction synthesis.
* **Embodied Policy:** [OpenPI](https://github.com/Physical-Intelligence/openpi) for real-world dual-arm execution & [StarVLA](https://github.com/starvla/starvla) for simulation.
* **Kinematic Layer:** Asynchronous producer–consumer control thread with receding horizon ($H=50$) and temporal Exponential Moving Average ($\beta=0.35$).

---

## 🧩 Module Guide

### ① Scene-aware-BCI-VLA — BCI Decoding

Implements the SSVEP-BCI pipeline: flicker stimulation, EEG acquisition, and frequency decoding.

| File | Role |
|---|---|
| `triggerBox.py` | Renders multi-frequency SSVEP flicker masks over candidate targets and synchronizes triggers with the EEG amplifier. |
| `fbcca.py` | Online FBCCA decoder that maps the selected SSVEP component to a target ID. |
| `interface.py` | Real-time acquisition interface for the Neuracle NeuSenW wireless EEG system. |
| `creat_raw_data.py` | Constructs and labels raw EEG datasets for offline analysis. |

### ② openpi — VLA Policy

A modified fork of [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) used to fine-tune and serve the **π0.5** policy for dual-arm manipulation. Key project-specific changes include the WebSocket policy serving stack (`src/openpi/serving/`), real-time-chunking (RTC) inference support, Piper/Aloha policy adaptations (`src/openpi/policies/`), and training configurations (`src/openpi/training/config.py`). See `openpi/docs/rtc_piper_deployment.md` for deployment notes.

### ③ Grounded-SAM-2 — Scene Perception

A modified fork of [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) (Florence-2 + SAM-2) that detects and segments candidate objects in the workspace and renders the SSVEP flicker masks onto them. `experiment.py` contains the scene-aware perception pipeline used in the paper.

### ④ scripts — System Integration

| Path | Role |
|---|---|
| `scripts/inference.py` | Closed-loop entry point running on the robot host: consumes BCI selections, queries scene context and the policy server, and drives the arms. |
| `scripts/server_api.py` | Hosts the fine-tuned π0.5 policy on the GPU workstation; receives synchronized multi-view RGB streams and language prompts, returns action chunks. |
| `scripts/robot/RTC_inference.py` | Latest robot execution client with receding-horizon control and asynchronous action smoothing. |
| `scripts/robot/SSVEP_inference.py` | SSVEP online-control variant with Gemini-based scene reasoning (`USE_LLM` ablation switch). |
| `scripts/bci/` | EEG streaming (`dataServer.py`), BCI mapping (`BCIMAP.py`), and decoding utilities on the robot side. |
| `scripts/training/` | Fine-tuning and normalization-statistics scripts (`train*.py`, `serve_policy.py`, `compute_norm_stats.py`). |
| `scripts/data_tools/` | Dataset conversion to LeRobot format (`convert_to_lerobot.py`, `ros_to_lerobot.py`) and inference-log duration analysis (`check.py`). |
| `scripts/docker/` | Dockerfile / compose files for reproducible policy serving. |

---

## 🛠️ Hardware Requirements

* **Dual-Arm Platform:** Agilex Piper 14-DoF dual-arm collaborative robot.
* **RGB-D Vision:** 3 Intel RealSense D435i cameras (1 overhead global view, 2 wrist-mounted views).
* **EEG Acquisition:** Neuracle NeuSenW wireless EEG system (Sampling rate: 1000 Hz; Channels: Fp1, Fp2, Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2).
* **Compute:** At least 48GB for inference and 100GB for fine-tuning.

---

## 📦 Installation & Setup

`openpi/` and `Grounded-SAM-2/` are already included in this repository, so a single clone is enough:

```bash
# 1. Create and activate a clean conda environment
conda create -n bci-vla python=3.10 -y
conda activate bci-vla

# 2. Install PyTorch with appropriate CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Clone the main repository (openpi & Grounded-SAM-2 are bundled inside)
git clone https://github.com/oyjt-hub/Scene-aware-BCI-VLA.git
cd Scene-aware-BCI-VLA

# 4. Install the two base frameworks
pip install -e ./openpi
pip install -e ./Grounded-SAM-2
```

The scene-reasoning module calls **Gemini-2.5-flash**; provide your key as an environment variable:

```bash
export GEMINI_API_KEY="your-gemini-api-key"   # required by SSVEP_inference.py / experiment.py
```

---

## 🚀 Running the System

The full pipeline runs as three cooperating processes:

| Process | Host | Command |
|---|---|---|
| 1. VLA policy server | GPU workstation | `python scripts/server_api.py` |
| 2. Scene-aware perception | Vision host | `python Grounded-SAM-2/experiment.py` (serves scene & masks on port 8000) |
| 3. BCI-VLA robot client | Robot host | `python scripts/inference.py` (or `python scripts/robot/RTC_inference.py` for receding-horizon control) |

Adjust `SERVER_HOST` / `SERVER_PORT` at the top of the client scripts to point at your policy and perception servers.

---

## 🧠 Adapting to Custom EEG Hardware

In our paper, online EEG decoding was validated using a 9-channel Neuracle (NeuSenW) wireless acquisition system at 1000 Hz.
Please adapt or write the real-time EEG streaming interface to match your specific EEG hardware SDK.

---

## 🙏 Acknowledgements

This project builds on the excellent open-source work of:

* [OpenPI](https://github.com/Physical-Intelligence/openpi) — π0.5 Vision-Language-Action models from Physical Intelligence.
* [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) — grounding & segmentation from IDEA-Research.
* [StarVLA](https://github.com/starvla/starvla) — simulation evaluation.
* [Neuracle](http://neuracle.cn/) — wireless EEG acquisition hardware.

## 📄 License

This project is released under the [Apache 2.0 License](LICENSE).
