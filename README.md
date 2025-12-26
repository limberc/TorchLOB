# PyTorchLOB: GPU-Accelerated Limit Order Book Simulator

**PyTorchLOB** is a PyTorch-based implementation of a Limit Order Book (LOB) simulator, designed for high-performance Reinforcement Learning (RL) in financial trading. It supports GPU acceleration, including **Mac MPS (Metal Performance Shaders)** for Apple Silicon and CUDA for NVIDIA GPUs.

## Key Features

*   **PyTorch Implementation**: Native `torch` tensors for easy integration with standard Deep RL libraries.
*   **Mac MPS Support**: Optimized to run efficiently on Apple M1/M2/M3 chips using the Metal backend.
*   **Real Market Data Replay**: Supports loading LOB data (e.g., Crypto CSVs) to drive realistic price dynamics.
*   **Deep Learning Models**: Includes **CNN (Price-Time)** and **Transformer** architectures for state-of-the-art LOB representation.
*   **Safe RL Agents**: Includes implementations of **PPO** with **PID-Lagrangian** constraints for safe execution handling (e.g., minimizing slippage).
*   **Gymnasium Compatible**: Fully compatible `gym.Env` interface (`TorchExecutionEnv`) for easy plugging into standard RL loops.

## Architecture

*   `torch_exchange`: Core package.
    *   `orderbook.py`: Vectorized Limit Order Book logic using PyTorch **JIT Compilation** for maximum speed.
    *   `environment.py`: Gym-compatible Execution Environment with **Dynamic Observation Spaces**.
    *   `ppo.py`: Customizable PPO Agent supporting **MLP**, **CNN**, and **Transformer** encoders.
    *   `models/networks.py`: Neural network definitions.

## Installation

### Prerequisites
*   Python 3.8+
*   PyTorch (with MPS support for Mac, or CUDA for Linux/Windows)

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

The best way to get started is to run the demo notebook:

```bash
demo.ipynb
```

This notebook demonstrates:

1.  **FIFO Baseline**: A standard passive execution strategy.
2.  **Real Data Replay**: Loading Bitcoin 1-sec LOB data to simulate realistic market conditions.
3.  **Safe RL (PPO)**: Training a PPO agent with slippage constraints on the LOB.

## Usage Example

### 1. Basic Setup
```python
import torch
from torch_exchange.environment import TorchExecutionEnv
from torch_exchange.ppo import PPOAgent

# Setup Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Create Environment (Mock Data)
env = TorchExecutionEnv(task='sell', task_size=5000, device=device, book_depth=10)

# Initialize Agent (MLP)
agent = PPOAgent(env, device=device, lr=3e-4, model_type='mlp')

# Train
agent.train(total_timesteps=10000)
```

### 2. Using Real Data & Advanced Models

Download the data from [Kaggle](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data).

`kaggle datasets download -d martinsn/high-frequency-crypto-limit-order-book-data`

```python
# Create Environment with Real Data
data_path = 'high-frequency-crypto-limit-order-book-data/BTC_1sec.csv'
env = TorchExecutionEnv(
    task='sell', 
    device=device, 
    data_path=data_path, 
    nrows=10000
)

# Initialize Agent with CNN (Frame Stacking)
agent = PPOAgent(
    env, 
    device=device, 
    model_type='cnn', 
    n_stack=4 # Stack 4 frames
)

# Train against real price history
agent.train(total_timesteps=50000)
```

## Acknowledgement

> This project is heavily modified from **[JAX-LOB](https://github.com/KangOxford/jax-lob)**. We acknowledge the original authors (Sascha Frey, Kang Li, et al. 2023) for their pioneering work in accelerating LOB simulations.

```
@misc{frey2023jaxlob,
      title={JAX-LOB: A GPU-Accelerated limit order book simulator to unlock large scale reinforcement learning for trading}, 
      author={Sascha Frey and Kang Li and Peer Nagy and Silvia Sapora and Chris Lu and Stefan Zohren and Jakob Foerster and Anisoara Calinescu},
      year={2023},
      eprint={2308.13289},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR}
}
```
