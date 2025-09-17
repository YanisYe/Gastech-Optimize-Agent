# GasTech Optimization

Deep reinforcement learning guided precise strategies for keeping multiple environmental impacts of agricultural within limits in China.

## Overview

This project implements a deep reinforcement learning approach to optimize agricultural gas technology strategies while maintaining environmental impact constraints in China. The system uses advanced RL algorithms based on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) to provide precise decision-making strategies for agricultural environmental management.

## Environment Setup

We recommend using `uv` for environment management:

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GasTech-Optimization
```

2. Create and activate virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

### Training

To train the reinforcement learning model for all agricultural regions:

```bash
python model/train.py
```

The training process will:
- Load agricultural and environmental data for each region
- Initialize the RL environment with curriculum learning
- Train PPO models using Stable-Baselines3
- Save model checkpoints and training logs for each region
- Use exponential learning rate scheduling

### Testing

To test the trained model for specific regions:

```bash
python model/test.py
```

The testing process will:
- Load the trained models for specified regions
- Evaluate performance using the trained policies
- Generate detailed results and visualizations
- Merge results from all regions into consolidated files
- Output environmental impact analysis

### Analysis

For detailed analysis of RL area technology impact:

```bash
python utils/analysis_rl_area_tech_impact.py
```

## Project Structure

```
├── model/                           # Model definitions and checkpoints
├── utils/                           # Utility functions and analysis tools
├── data/                            # Dataset and configuration files
├── results/                         # Output results and visualizations
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project configuration
└── README.md                        # This file
```

## Dependencies

Key dependencies include:
- PyTorch (2.7.1+)
- Gymnasium (0.29.0)
- NumPy (2.2.6)
- Pandas (2.2.3)
- Matplotlib (3.10.3)
- SciPy (1.15.3)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms

## License

MIT License
