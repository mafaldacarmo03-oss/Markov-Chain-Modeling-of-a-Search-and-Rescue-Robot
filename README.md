# Rescue Robot Markov Chain App

This repository contains a Streamlit application for modeling a simplified search-and-rescue robot in a hazardous grid environment using Markov chains.

The app studies the same mission with two stochastic models:

1. **Discrete-time Markov chain (DTMC):** the robot moves step by step on a grid. Each cell is a Markov state, hazards and the victim are absorbing states, and uncertainty is introduced through the probability of executing the intended movement.
2. **Continuous-time Markov chain (CTMC):** the same transition structure is converted into a generator matrix so that mission evolution can be analyzed over continuous time.

The app is inspired by the FrozenLake interpretation: the agent is a rescue robot, safe tiles are traversable terrain, holes are hazards, and the goal is the victim or rescue target. The current implementation builds and analyzes the grid directly instead of requiring Gymnasium at runtime.

## Repository structure

```text
rescue_markov_app/
├── app.py
├── requirements.txt
├── utils/
│   ├── discrete.py
│   ├── continuous.py
│   ├── plotting.py
│   └── policies.py
└── README.md
```

## Main features

- Random generation of 4x4 or 8x8 rescue grids.
- Choice between Random and Greedy policies.
- Adjustable probability of intended movement.
- Transition graph visualization with PyVis.
- One-step transition probability visualization.
- Transition matrix heatmaps.
- n-step prediction using pi_n = pi_0 P^n.
- Sampled trajectory animation.
- Markov-chain class classification.
- Absorbing-chain analysis with Q, R, N=(I-Q)^(-1), expected absorption time, and absorption probabilities B=NR.
- Continuous-time generator matrix construction.
- Continuous-time transition matrix P(t)=exp(Qt).
- Animated CTMC probability evolution.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the app

From the repository folder:

```bash
streamlit run app.py
```

The app should open automatically in the browser. If it does not, open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

## Notes

This is an educational stochastic-modeling project. It does not simulate full robot dynamics, continuous physical motion, sensors, mapping, localization, or battery consumption. The objective is to provide a clear visual and mathematical demonstration of Markov chains in uncertain rescue navigation.
