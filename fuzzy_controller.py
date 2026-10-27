"""Mamdani fuzzy traffic-signal controller for SmartFlow AI.

Inputs:
  density: normalized lane occupancy/congestion, 0..100 percent
  waiting_time: waiting time of the queue, 0..120 seconds

Output:
  green_seconds: recommended green duration, 10..90 seconds

This is an educational simulation model. Tune membership functions and rules
using repeatable tests before making performance claims.
"""
import numpy as np


def triangular(x, a, b, c):
    """Triangular membership function with support [a, c] and peak b."""
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    if b != a:
        mask = (x >= a) & (x <= b)
        result[mask] = (x[mask] - a) / (b - a)
    else:
        result[x == b] = 1.0
    if c != b:
        mask = (x >= b) & (x <= c)
        result[mask] = np.maximum(result[mask], (c - x[mask]) / (c - b))
    else:
        result[x == b] = 1.0
    return np.clip(result, 0.0, 1.0)


def left_shoulder(x, a, b):
    """Membership is 1 up to a, then decreases to 0 at b."""
    x = np.asarray(x, dtype=float)
    return np.where(x <= a, 1.0, np.where(x >= b, 0.0, (b - x) / (b - a)))


def right_shoulder(x, a, b):
    """Membership is 0 up to a, then increases to 1 at b."""
    x = np.asarray(x, dtype=float)
    return np.where(x <= a, 0.0, np.where(x >= b, 1.0, (x - a) / (b - a)))


def density_memberships(density):
    return {
        "Low": float(left_shoulder(density, 20, 50)),
        "Medium": float(triangular(np.array([density]), 25, 50, 75)[0]),
        "High": float(right_shoulder(density, 50, 80)),
    }


def waiting_memberships(waiting_time):
    return {
        "Short": float(left_shoulder(waiting_time, 15, 45)),
        "Medium": float(triangular(np.array([waiting_time]), 25, 60, 95)[0]),
        "Long": float(right_shoulder(waiting_time, 75, 105)),
    }


def output_memberships(output_universe):
    return {
        "Short": left_shoulder(output_universe, 10, 35),
        "Medium": triangular(output_universe, 25, 50, 75),
        "Long": right_shoulder(output_universe, 60, 90),
    }


# Rows are density categories; columns are waiting-time categories.
# Each cell returns the consequent green-duration category.
RULE_TABLE = {
    ("Low", "Short"): "Short",
    ("Low", "Medium"): "Short",
    ("Low", "Long"): "Medium",
    ("Medium", "Short"): "Short",
    ("Medium", "Medium"): "Medium",
    ("Medium", "Long"): "Long",
    ("High", "Short"): "Medium",
    ("High", "Medium"): "Long",
    ("High", "Long"): "Long",
}


def infer_green_duration(density, waiting_time):
    """Return (green_seconds, details) using Mamdani min-max inference.

    The AND operator is min, rule aggregation is max, and defuzzification
    uses the centroid of area.
    """
    density = float(np.clip(density, 0, 100))
    waiting_time = float(np.clip(waiting_time, 0, 120))

    d_members = density_memberships(density)
    w_members = waiting_memberships(waiting_time)

    universe = np.linspace(10, 90, 801)
    output_sets = output_memberships(universe)
    aggregated = np.zeros_like(universe)
    fired_rules = []

    for (d_label, w_label), consequent in RULE_TABLE.items():
        strength = min(d_members[d_label], w_members[w_label])
        if strength > 0:
            clipped_consequent = np.minimum(strength, output_sets[consequent])
            aggregated = np.maximum(aggregated, clipped_consequent)
            fired_rules.append({
                "rule": f"IF density is {d_label} AND waiting time is {w_label} THEN green duration is {consequent}",
                "strength": round(float(strength), 3),
                "output": consequent,
            })

    area = float(np.sum(aggregated))
    if area == 0:
        # Safe fallback; should not occur with the defined fuzzy sets.
        green_seconds = 50.0
    else:
        green_seconds = float(np.sum(universe * aggregated) / area)

    details = {
        "density_memberships": {k: round(v, 3) for k, v in d_members.items()},
        "waiting_memberships": {k: round(v, 3) for k, v in w_members.items()},
        "fired_rules": sorted(fired_rules, key=lambda r: r["strength"], reverse=True),
        "output_universe": universe,
        "aggregated_output": aggregated,
    }
    return round(green_seconds, 1), details
