# Trafficient

An intelligent traffic control system using Deep Q-Learning to optimize traffic signal timing. This project achieved a 42.4% reduction in average wait times and a 34.9% increase in traffic flow speed.

## Overview

Trafficient uses reinforcement learning to dynamically control traffic signals, adapting to real-time traffic conditions. The system is built on SUMO (Simulation of Urban MObility) and PyTorch, implementing a Deep Q-Network (DQN) for intelligent decision-making.

## Key Features

- Real-time traffic signal optimization
- Predictive queue management
- Dynamic phase timing
- Multi-intersection support
- Performance metrics tracking

## Requirements

- Python 3.8+
- SUMO (Simulation of Urban MObility)
- PyTorch
- NumPy
- Matplotlib

## Installation

1. Install SUMO:
```bash
# Set SUMO_HOME environment variable
export SUMO_HOME="path/to/sumo"
```

2. Install Python dependencies:
```bash
pip install torch numpy matplotlib
```

3. Clone the repository:
```bash
git clone [repository-url]
cd trafficient
```

## Usage

1. Run the reinforcement learning model:
```bash
python rl_traffic_control.py
```

2. Compare with traditional methods:
```bash
python compare_methods.py
```

3. Generate random traffic patterns:
```bash
python random_trips.py
```

## Project Structure

```
trafficient/
├── rl_traffic_control.py     # Main RL implementation
├── traditional_traffic.net.xml    # Network configuration
├── traditional_traffic.sumo.cfg   # SUMO configuration
├── random_trips.py           # Traffic generation
├── traditional_traffic_control.py # Traditional Traffic control
├── random_traffic.rou.xml # Traditional Traffic control
└── compare_methods.py        # Performance comparison


```

## Results

- Average wait time reduction: 50% 
- Traffic flow speed increase: 34.9%
- Improved throughput across 8+ traffic approaches

## Acknowledgments

- SUMO Traffic Simulation
- PyTorch Team
- Traffic Control Research Community
