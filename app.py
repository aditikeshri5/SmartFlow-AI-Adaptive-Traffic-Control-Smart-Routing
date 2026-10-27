
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from fuzzy_controller import FuzzyTrafficController
from traffic_simulator import TrafficSimulator


st.set_page_config(
    page_title="SmartFlow AI",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 SmartFlow AI")
st.subheader("Adaptive Traffic Control & Smart Routing")
st.write(
    "An Artificial Intelligence & Soft Computing project using "
    "Mamdani Fuzzy Logic to adapt traffic signal durations."
)

st.divider()

# --------------------------------------------------
# SECTION 1: Interactive fuzzy controller
# --------------------------------------------------

st.header("1. Fuzzy Traffic Controller")
st.write(
    "Adjust traffic density and queue waiting time to see "
    "how fuzzy logic determines the green-light duration."
)

controller = FuzzyTrafficController()

col1, col2 = st.columns(2)

with col1:
    density = st.slider(
        "Traffic density / occupancy score (%)",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
    )

with col2:
    waiting_time = st.slider(
        "Queue waiting time (seconds)",
        min_value=0,
        max_value=120,
        value=60,
        step=5,
    )

result = controller.calculate_green_time(
    density=density,
    waiting_time=waiting_time,
)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Green-light duration", f"{result['green_time']} s")

with metric_col2:
    st.metric("Traffic density", f"{density}%")

with metric_col3:
    st.metric("Queue waiting time", f"{waiting_time} s")

st.caption(
    "Density is a normalized occupancy/congestion score, not a "
    "measured road density in vehicles per kilometre."
)

# --------------------------------------------------
# SECTION 2: Membership functions
# --------------------------------------------------

st.subheader("Fuzzy Membership Functions")

tab1, tab2, tab3 = st.tabs(
    ["Traffic density", "Waiting time", "Green duration"]
)

with tab1:
    fig, ax = plt.subplots()
    x = controller.density_range

    for label, points in {
        "Low": [0, 0, 50],
        "Medium": [20, 50, 80],
        "High": [50, 100, 100],
    }.items():
        ax.plot(
            x,
            controller.trimf(x, points),
            label=label,
        )

    ax.axvline(density, linestyle="--", label="Current input")
    ax.set_xlabel("Density score (%)")
    ax.set_ylabel("Membership degree")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig, ax = plt.subplots()
    x = controller.waiting_range

    for label, points in {
        "Short": [0, 0, 60],
        "Medium": [20, 60, 100],
        "Long": [60, 120, 120],
    }.items():
        ax.plot(
            x,
            controller.trimf(x, points),
            label=label,
        )

    ax.axvline(waiting_time, linestyle="--", label="Current input")
    ax.set_xlabel("Waiting time (seconds)")
    ax.set_ylabel("Membership degree")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fig, ax = plt.subplots()
    x = controller.green_range

    for label, points in {
        "Short": [10, 10, 30],
        "Medium": [20, 35, 50],
        "Long": [40, 60, 60],
    }.items():
        ax.plot(
            x,
            controller.trimf(x, points),
            label=label,
        )

    ax.axvline(
        result["green_time"],
        linestyle="--",
        label="Defuzzified output",
    )
    ax.set_xlabel("Green duration (seconds)")
    ax.set_ylabel("Membership degree")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# SECTION 3: Fired rules and aggregated output
# --------------------------------------------------

st.subheader("Fuzzy Rule Evaluation")

if result["fired_rules"]:
    rules_df = pd.DataFrame(result["fired_rules"])
    rules_df.columns = [
        "Density category",
        "Waiting category",
        "Output category",
        "Rule strength",
    ]
    st.dataframe(rules_df, use_container_width=True)
else:
    st.info("No rules fired for these inputs.")

fig, ax = plt.subplots()

ax.plot(
    result["green_range"],
    result["aggregated_output"],
    label="Aggregated fuzzy output",
)

ax.axvline(
    result["green_time"],
    linestyle="--",
    label=f"Centroid = {result['green_time']} s",
)

ax.set_xlabel("Green duration (seconds)")
ax.set_ylabel("Membership degree")
ax.set_ylim(0, 1.05)
ax.set_title("Mamdani Aggregation and Centroid Defuzzification")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.divider()

# --------------------------------------------------
# SECTION 4: Traffic simulation
# --------------------------------------------------

st.header("2. Traffic Simulation")
st.write(
    "Compare fixed-time traffic signals with fuzzy-controlled "
    "signals using the same arrival pattern."
)

with st.sidebar:
    st.header("Simulation Settings")

    simulation_minutes = st.slider(
        "Simulation duration (minutes)",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
    )

    fixed_green = st.slider(
        "Fixed green duration (seconds)",
        min_value=10,
        max_value=60,
        value=30,
        step=5,
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=10000,
        value=42,
        step=1,
    )

    run_button = st.button(
        "Run Traffic Comparison",
        type="primary",
        use_container_width=True,
    )

if run_button:
    duration_seconds = simulation_minutes * 60

    with st.spinner("Simulating traffic..."):
        fixed_sim = TrafficSimulator(
            simulation_time=duration_seconds,
            seed=int(seed),
            fixed_green=fixed_green,
        )

        fuzzy_sim = TrafficSimulator(
            simulation_time=duration_seconds,
            seed=int(seed),
        )

        fixed_results = fixed_sim.run_simulation("fixed")
        fuzzy_results = fuzzy_sim.run_simulation("fuzzy")

    st.session_state["fixed_results"] = fixed_results
    st.session_state["fuzzy_results"] = fuzzy_results

if (
    "fixed_results" in st.session_state
    and "fuzzy_results" in st.session_state
):
    fixed = st.session_state["fixed_results"]
    fuzzy = st.session_state["fuzzy_results"]

    st.subheader("Performance Comparison")

    metrics = [
        ("Average waiting time (s)", "average_wait", "lower"),
        ("Vehicles served", "vehicles_served", "higher"),
        ("Vehicles remaining", "remaining_vehicles", "lower"),
        ("Maximum combined queue", "max_queue", "lower"),
        ("Throughput (vehicles/min)", "throughput_per_minute", "higher"),
    ]

    for start in range(0, len(metrics), 3):
        columns = st.columns(3)

        for col, item in zip(columns, metrics[start:start + 3]):
            label, key, preference = item

            with col:
                st.markdown(f"**{label}**")
                st.metric("Fixed-time", fixed[key])
                st.metric(
                    "Fuzzy control",
                    fuzzy[key],
                    delta=round(fuzzy[key] - fixed[key], 2),
                    delta_color=(
                        "inverse" if preference == "lower" else "normal"
                    ),
                )

    st.subheader("Queue Length Over Time")

    fixed_queue = pd.DataFrame(fixed["queue_history"])
    fuzzy_queue = pd.DataFrame(fuzzy["queue_history"])

    approach = st.selectbox(
        "Select an approach",
        TrafficSimulator.APPROACHES,
    )

    fig, ax = plt.subplots()

    ax.plot(
        fixed_queue["time"] / 60,
        fixed_queue[approach],
        label="Fixed-time",
    )

    ax.plot(
        fuzzy_queue["time"] / 60,
        fuzzy_queue[approach],
        label="Fuzzy control",
    )

    ax.set_xlabel("Simulation time (minutes)")
    ax.set_ylabel("Vehicles waiting")
    ax.set_title(f"{approach} Approach Queue")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Green Signal Allocation")

    green_data = pd.DataFrame(
        fuzzy["green_time_history"]
    )

    if not green_data.empty:
        green_data["time_minutes"] = green_data["time"] / 60

        fig, ax = plt.subplots()

        for approach_name, group in green_data.groupby("approach"):
            ax.scatter(
                group["time_minutes"],
                group["green_seconds"],
                label=approach_name,
            )

        ax.set_xlabel("Simulation time (minutes)")
        ax.set_ylabel("Green duration (seconds)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

else:
    st.info(
        "Set your simulation parameters in the sidebar, "
        "then click **Run Traffic Comparison**."
    )

st.divider()

st.header("3. Project Roadmap")
st.markdown(
    """
    - **Implemented here:** Mamdani fuzzy inference and a four-way
      intersection simulation.
    - **Next extension:** Model a road network and use A* search to
      find routes using estimated driving time, signal delay, and queue delay.
    - **Evaluation:** Test multiple random seeds and traffic arrival patterns
      before drawing conclusions about performance.
    """
)

st.caption(
    "Academic simulation only. Results depend on the model assumptions "
    "and do not represent live road traffic."
)
