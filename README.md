# SmartFlow AI — Starter Prototype

A student project for Artificial Intelligence & Soft Computing (AISC).

## Current stage

This first milestone implements a **Mamdani Fuzzy Inference System** to recommend
a green-light duration based on:

- Traffic density: 0–100% normalized lane occupancy/congestion score
- Queue waiting time: 0–120 seconds

It uses:
- Fuzzification with Low/Medium/High and Short/Medium/Long sets
- A 3×3 IF–THEN rule base
- Minimum for AND
- Maximum for rule aggregation
- Centroid defuzzification

This is a simulation prototype, not a real-world traffic controller. The
membership functions and rules are initial design choices and should be tuned
and evaluated through repeatable experiments.

## Run locally

1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

## Files

- `fuzzy_controller.py` — membership functions, rule base, inference and centroid defuzzification
- `app.py` — interactive Streamlit UI and plots
- `requirements.txt` — dependencies

## Planned next steps

1. Add a time-stepped traffic simulator with vehicle arrivals and queue discharge.
2. Compare fuzzy signal timing with a fixed-time baseline under identical traffic scenarios.
3. Add a road graph and A* route planning using estimated travel time and signal/queue delays.
4. Recalculate routes when traffic costs change and evaluate whether congestion metrics improve.

## Important modeling note

The current density input is a normalized occupancy/congestion score, not the
formal traffic-engineering density measured in vehicles per kilometre. For a
more physically meaningful model, define lane length, vehicle spacing and
storage capacity, then document the chosen assumptions.
