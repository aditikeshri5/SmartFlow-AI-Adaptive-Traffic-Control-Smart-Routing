import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from fuzzy_controller import (
    infer_green_duration,
    density_memberships,
    waiting_memberships,
    left_shoulder,
    triangular,
    right_shoulder,
)

st.set_page_config(page_title="SmartFlow AI", page_icon="🚦", layout="wide")
st.title("🚦 SmartFlow AI")
st.subheader("Fuzzy Logic-Based Intelligent Traffic Signal Control")
st.caption("Phase 1: Mamdani fuzzy controller prototype — no real traffic data or hardware required.")

st.info(
    "This prototype uses simulated inputs. Traffic density is modeled as a "
    "normalized lane-occupancy/congestion score (0–100%), not vehicles per kilometre."
)

with st.sidebar:
    st.header("Traffic conditions")
    density = st.slider("Traffic density (%)", min_value=0, max_value=100, value=70, step=1)
    waiting_time = st.slider("Queue waiting time (seconds)", min_value=0, max_value=120, value=65, step=1)
    st.markdown("---")
    st.caption("The values are illustrative. We'll tune the membership functions and rules after testing.")

green, details = infer_green_duration(density, waiting_time)

col1, col2, col3 = st.columns(3)
col1.metric("Traffic density", f"{density}%")
col2.metric("Waiting time", f"{waiting_time} s")
col3.metric("Recommended green duration", f"{green} s")

st.markdown("### 1. Fuzzification")
left, right = st.columns(2)

with left:
    st.markdown("**Traffic density membership**")
    x = np.linspace(0, 100, 501)
    fig, ax = plt.subplots()
    ax.plot(x, left_shoulder(x, 20, 50), label="Low")
    ax.plot(x, triangular(x, 25, 50, 75), label="Medium")
    ax.plot(x, right_shoulder(x, 50, 80), label="High")
    ax.axvline(density, linestyle="--", label=f"Current = {density}%")
    ax.set(xlabel="Density (%)", ylabel="Membership degree", ylim=(-0.03, 1.05))
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.write(details["density_memberships"])

with right:
    st.markdown("**Waiting time membership**")
    x = np.linspace(0, 120, 601)
    fig, ax = plt.subplots()
    ax.plot(x, left_shoulder(x, 15, 45), label="Short")
    ax.plot(x, triangular(x, 25, 60, 95), label="Medium")
    ax.plot(x, right_shoulder(x, 75, 105), label="Long")
    ax.axvline(waiting_time, linestyle="--", label=f"Current = {waiting_time} s")
    ax.set(xlabel="Waiting time (seconds)", ylabel="Membership degree", ylim=(-0.03, 1.05))
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.write(details["waiting_memberships"])

st.markdown("### 2. Rule evaluation")
if details["fired_rules"]:
    st.dataframe(
        [
            {"Rule": r["rule"], "Firing strength": r["strength"], "Output category": r["output"]}
            for r in details["fired_rules"]
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("No rule fired. Check membership-function definitions.")

st.markdown("### 3. Aggregation and centroid defuzzification")
fig, ax = plt.subplots()
ax.fill_between(
    details["output_universe"],
    details["aggregated_output"],
    alpha=0.35,
    label="Aggregated fuzzy output",
)
ax.plot(details["output_universe"], details["aggregated_output"], label="Aggregated output")
ax.axvline(green, linestyle="--", label=f"Centroid = {green} s")
ax.set(xlabel="Green-light duration (seconds)", ylabel="Membership degree", ylim=(-0.03, 1.05))
ax.grid(alpha=0.25)
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.markdown("### 4. Current rule interpretation")
top_rules = details["fired_rules"][:3]
if top_rules:
    for rule in top_rules:
        st.write(f"- **Strength {rule['strength']:.3f}:** {rule['rule']}")

st.warning(
    "Prototype limitation: this stage calculates a green duration from fuzzy rules. "
    "It does not yet simulate vehicle arrivals, queue discharge, competing approaches, "
    "or signal safety constraints. Those are the next implementation stage."
)
