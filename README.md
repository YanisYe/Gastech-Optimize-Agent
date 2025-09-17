# 🌾 GasTech Optimization

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1+-red.svg)](https://pytorch.org)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-Latest-green.svg)](https://github.com/DLR-RM/stable-baselines3)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> This repository is the implementation of the paper: **"Deep reinforcement learning guided precise strategies for keeping multiple environmental impacts of agricultural within limits in China"**

## 📋 Overview

This project implements a deep reinforcement learning approach to optimize agricultural gas technology strategies while maintaining environmental impact constraints in China. The system uses advanced RL algorithms based on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) to provide precise decision-making strategies for agricultural environmental management.

### 🎯 Key Features

- 🤖 **Deep Reinforcement Learning**: PPO-based optimization using Stable-Baselines3
- 🌱 **Agricultural Focus**: Specialized for Chinese agricultural environmental management
- 📊 **Curriculum Learning**: Progressive training strategy for better convergence
- 📈 **Comprehensive Analysis**: Detailed environmental impact assessment tools

## 🚀 Environment Setup

We recommend using `uv` for environment management:

### 📋 Prerequisites
- 🐍 Python 3.12+
- 📦 [uv](https://docs.astral.sh/uv/) package manager

### ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YanisYe/Gastech-Optimize-Agent.git
cd Gastech-Optimize-Agent
```

2. **Create and activate virtual environment with uv:**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
uv pip install -r requirements.txt
```

## 🎮 Usage

### 🏋️ Training

To train the reinforcement learning model for all agricultural regions:

```bash
python model/train.py
```

**The training process will:**
- 📊 Load agricultural and environmental data for each region
- 🎯 Initialize the RL environment with curriculum learning
- 🤖 Train PPO models using Stable-Baselines3
- 💾 Save model checkpoints and training logs for each region
- 📈 Use exponential learning rate scheduling

### 🧪 Testing

To test the trained model for specific regions:

```bash
python model/test.py
```

**The testing process will:**
- 🔄 Load the trained models for specified regions
- 📊 Evaluate performance using the trained policies
- 📈 Generate detailed results and visualizations
- 🔗 Merge results from all regions into consolidated files
- 🌍 Output environmental impact analysis

### 📊 Analysis

For detailed analysis of RL area technology impact:

```bash
python utils/analysis_rl_area_tech_impact.py
```

## 📁 Project Structure

```
├── 📂 model/                        # Model definitions and checkpoints
│   ├── train.py                     # Training script
│   ├── test.py                      # Testing script
│   ├── GasEnviroment_curriculum_learning.py  # RL environment
│   └── ...
├── 📂 utils/                        # Utility functions and analysis tools
├── 📂 data/                         # Dataset and configuration files
├── 📂 results/                      # Output results and visualizations
├── 📂 stable-baselines3/            # Custom Stable-Baselines3 implementation
├── 📄 requirements.txt              # Python dependencies
├── 📄 pyproject.toml                # Project configuration
└── 📄 README.md                     # This file
```

## 📦 Dependencies

**Key dependencies include:**
- 🤖 **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** - RL algorithms

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**🌾 GasTech Optimization** - *Optimizing agricultural environmental management through deep reinforcement learning*

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/YanisYe/Gastech-Optimize-Agent)
[![Paper](https://img.shields.io/badge/📄-Paper-blue?style=for-the-badge)](https://github.com/YanisYe/Gastech-Optimize-Agent)

</div>
