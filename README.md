# PyTorchLOB: GPU-Accelerated Limit Order Book Simulator

**PyTorchLOB** is a PyTorch-based implementation of a Limit Order Book (LOB) simulator, designed for high-performance Reinforcement Learning (RL) in financial trading. It supports GPU acceleration, including **Mac MPS (Metal Performance Shaders)** for Apple Silicon and CUDA for NVIDIA GPUs.

> **Acknowledgement**: This project is natively based on and inspired by **[JAX-LOB](https://github.com/KangOxford/jax-lob)**. We acknowledge the original authors (Sascha Frey, Kang Li, et al.) for their pioneering work in accelerating LOB simulations.

## Key Features

*   **PyTorch Implementation**: Native `torch` tensors for easy integration with standard Deep RL libraries.
*   **Mac MPS Support**: Optimized to run efficiently on Apple M1/M2/M3 chips using the Metal backend.
*   **Safe RL Agents**: Includes implementations of **PPO** with **PID-Lagrangian** constraints for safe execution handling (e.g., minimizing slippage).
*   **Gymnasium Compatible**: Fully compatible `gym.Env` interface (`TorchExecutionEnv`) for easy plugging into standard RL loops.

## Architecture

*   `torch_exchange`: Core package.
    *   `orderbook.py`: Vectorized Limit Order Book logic using PyTorch.
    *   `environment.py`: Gym-compatible Execution Environment.
    *   `ppo.py`: Customizable PPO Agent with Safe RL (PID-Lagrangian) capabilities.

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

This notebook demonstrates:

1.  **FIFO Baseline**: A standard passive execution strategy.
2.  **Safe RL (PPO)**: Training a PPO agent with slippage constraints on the LOB.

## Usage Example

```python
import torch
from torch_exchange.environment import TorchExecutionEnv
from torch_exchange.ppo import PPOAgent

# 1. Setup Device (Mac MPS or CUDA)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 2. Create Environment
# task='sell': Execute a sell order of 5000 units
env = TorchExecutionEnv(task='sell', task_size=5000, device=device, book_depth=10, tick_size=100)

# 3. Initialize Agent
agent = PPOAgent(env, device=device, lr=3e-4, cost_limit=0.5)

# 4. Train
agent.train(total_timesteps=10000)
```

## Citation

If you use this work or the original logic, please cite the original JAX-LOB paper:

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
