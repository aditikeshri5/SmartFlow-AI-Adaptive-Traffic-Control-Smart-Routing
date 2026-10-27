
# 🚦 SmartFlow AI — Adaptive Traffic Control & Smart Routing

SmartFlow AI is an Artificial Intelligence & Soft Computing project that
uses **Mamdani Fuzzy Logic** to adapt traffic signal durations according
to traffic congestion and vehicle waiting time.

The project simulates a four-way intersection and compares fuzzy-controlled
signals with conventional fixed-time signals. It is designed to explore
how intelligent traffic control can support congestion-aware transportation.

## 🎯 Objectives

- Implement a Mamdani Fuzzy Inference System.
- Model traffic density and waiting time using fuzzy membership functions.
- Dynamically determine green-light duration using fuzzy rules.
- Simulate traffic queues at a four-way intersection.
- Compare fixed-time and fuzzy-controlled signal strategies.
- Extend the system with A* search for congestion-aware route planning.

## 🧠 Artificial Intelligence Techniques

### 1. Fuzzy Logic

The controller uses two inputs:

- **Traffic density:** A normalized occupancy/congestion score from 0–100%.
- **Waiting time:** Queue waiting time from 0–120 seconds.

The output is the green-light duration, ranging from 10–60 seconds.

Fuzzy categories include:

| Variable | Categories |
|---|---|
| Traffic density | Low, Medium, High |
| Waiting time | Short, Medium, Long |
| Green duration | Short, Medium, Long |

The controller applies nine fuzzy rules using Mamdani inference,
min for rule conjunction, max for aggregation, and centroid
defuzzification.

Example rule:

> IF traffic density is HIGH AND waiting time is LONG,
> THEN green-light duration is LONG.

### 2. Traffic Simulation

The simulator models:
- Four traffic approaches: North, East, South, and West.
- Random vehicle arrivals.
- Vehicle queues and signal phases.
- Fixed-time and fuzzy-controlled signal strategies.

### 3. Planned A* Route Optimization

The next development phase will represent roads as a graph and use
A* search to identify routes based on estimated travel costs.

The planned road cost is:

`Road Cost = Driving Time + Signal Delay + Queue Delay`

A* evaluates paths using:

`f(n) = g(n) + h(n)`

where `g(n)` is the accumulated route cost and `h(n)` is the estimated
remaining cost to the destination.

## 🛠️ Technology Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Streamlit

## 📁 Project Structure

```text
SmartFlow_AI/
├── app.py
├── fuzzy_controller.py
├── traffic_simulator.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later recommended
- pip
- VS Code (optional)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
streamlit run app.py
```

The dashboard should open in your browser.

## 📊 Evaluation Metrics

The current dashboard reports:

- Average waiting time of served vehicles
- Number of vehicles served
- Number of vehicles remaining
- Maximum combined queue observed
- Throughput in vehicles per minute

Results depend on simulation settings, random arrivals, and model
assumptions. Multiple random seeds and traffic patterns should be
tested before drawing conclusions about performance.

## ⚠️ Current Limitations

- Traffic is simulated rather than collected from live roads.
- Density is represented by a normalized queue occupancy score.
- Vehicle movement, turning conflicts, pedestrians, and road geometry
  are simplified.
- The current implementation does not yet include A* routing.
- Simulation results do not guarantee improvements in real-world traffic.

## 🚀 Future Enhancements

- A* congestion-aware route optimization.
- Dynamic road-network modelling.
- Multiple traffic demand scenarios.
- Repeated experiments with different random seeds.
- Comparative charts and statistical evaluation.
- Optional live traffic data integration in a future version.

## 👩‍💻 Project

**SmartFlow AI — Adaptive Traffic Control & Smart Routing**

An academic project exploring fuzzy logic and intelligent routing
for traffic management.
